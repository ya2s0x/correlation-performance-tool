import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
from datetime import timedelta
from matplotlib.backends.backend_pdf import PdfPages


# 🎨 Charte graphique
primary_color = "#4E26DF"
secondary_color = "#7CEF17"
heatmap_colors = ["#B8A8F2", "#C1E5F5", "#C3F793"]
performance_colors = ["#4E26DF", "#7CEF17", "#35434B", "#B8A8F2", "#C1E5F5", "#C3F793", "#F2CFEE","#F2F2F2","#FCD9C4", "#A7C7E7", "#D4C2FC", "#F9F6B2", "#C4FCD2"]

# 📌 Mapping des actifs
asset_mapping = {
    "MSCI World": "URTH",
    "Nasdaq": "^IXIC",
    "S&P 500": "^GSPC",
    "US 10Y Yield": "^TNX",
    "Dollar Index": "DX-Y.NYB",
    "Gold": "GC=F",
    "iShares Bonds Agregate":"AGGG.L"
}

# Début configuration des portefeuilles
portfolio_allocations = {
    "Portfolio 1": {
        "^GSPC": 0.60,
        "AGGG.L": 0.40
    },
    "Portfolio 2": {
        "^GSPC": 0.57,       # 95% * 60%
        "AGGG.L": 0.38,     # 95% * 40%
        "GC=F": 0.05        # 5% Gold
    }
}

crypto_mapping = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Cardano (ADA)": "ADA-USD",
    "Ripple (XRP)": "XRP-USD",
    "Polkadot (DOT)": "DOT-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Stellar (XLM)": "XLM-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Polygon (MATIC)": "MATIC-USD",
    "Cosmos (ATOM)": "ATOM-USD",
    "Algorand (ALGO)": "ALGO-USD",
    "Filecoin (FIL)": "FIL-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "Tron (TRX)": "TRX-USD",
    "Sui (SUI)": "SUI20947-USD",
    "Bitcoin Cash (BCH)": "BCH-USD",
    "Hyperliquid (HYPE)": "HYPE32196-USD",
    "Monero (XMR)": "XMR-USD",
    "Uniswap (UNI)": "UNI7083-USD",
    "Near Protocol (NEAR)": "NEAR-USD",
    "Ondo (ONDO)": "ONDO-USD",
    "Aave (AAVE)": "AAVE-USD",
    "Cronos (CRO)": "CRO-USD",
    "Vechain (VET)": "VET-USD",
    "Bittensor (TAO)": "TAO22974-USD",
    "Celestia (TIA)": "TIA-USD",
    "Arbitrum (ARB)": "ARB11841-USD",
    "Render (RNDR)": "RENDER-USD",
    "Optimism (OP)": "OP-USD",
    "Fetch.AI (FET)": "FET-USD",
    "Ethena (ENA)": "ENA-USD",
    "Maker (MKR)": "MKR-USD",
    "Jupiter (JUP)": "JUP29210-USD",
    "Synthetix (SNX)": "SNX-USD",
    "Flux (FLUX)": "FLUX-USD"
}

us_equity_mapping = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Alphabet (GOOGL)": "GOOGL",
    "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA"
}

full_asset_mapping = {**asset_mapping, **crypto_mapping, **us_equity_mapping}
asset_names_map = {v: k for k, v in full_asset_mapping.items()}


# 🖥️ Interface
st.set_page_config(page_title="Alphacap", layout="wide")
st.title("Comparaison de performances d'actifs")

# Interface dynamique de portefeuille crypto
st.markdown("## 💼 Composition du portefeuille crypto")

crypto_allocation = []
crypto_options = list(crypto_mapping.keys())

crypto_global_pct = st.number_input("% du portefeuille total alloué à l'allocation crypto", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
num_crypto = st.number_input("Nombre d'actifs cryptoactifs dans la poche", min_value=1, max_value=15, step=1, value=1)

total_pct = 0
for i in range(num_crypto):
    cols = st.columns([3, 1])
    with cols[0]:
        selected_crypto = st.selectbox(f"Crypto {i+1}", crypto_options, key=f"crypto_{i}")
    with cols[1]:
        pct = st.number_input(f"% de la crypto {i+1} dans la poche", min_value=0.0, max_value=100.0, step=0.1, key=f"pct_{i}")

    crypto_allocation.append((selected_crypto, pct))
    total_pct += pct

if total_pct != 100:
    st.warning(f"⚠️ La somme des pourcentages de la poche crypto est {total_pct:.1f}%. Elle doit être exactement 100%.")
elif crypto_global_pct <= 0:
    st.warning("⚠️ Le pourcentage global alloué à la poche crypto doit être supérieur à 0.")
else:
    st.success("✅ Répartition valide du portefeuille.")

    portfolio3 = {}
    classic_weight = 1 - crypto_global_pct / 100

    for ticker, weight in portfolio_allocations["Portfolio 1"].items():
        portfolio3[ticker] = weight * classic_weight

    for name, pct in crypto_allocation:
        ticker = crypto_mapping[name]
        weight = (pct / 100) * (crypto_global_pct / 100)
        portfolio3[ticker] = portfolio3.get(ticker, 0) + weight

    portfolio_allocations["Portfolio 3"] = portfolio3

    # 🔢 Fonction de calcul des métriques sur un portefeuille donné

def compute_portfolio_metrics(prices, allocations, reference_returns=None):
    weights = np.array([allocations[ticker] for ticker in allocations if ticker in prices.columns])
    tickers = [ticker for ticker in allocations if ticker in prices.columns]
    data = prices[tickers].dropna()
    returns = data.pct_change().dropna()

    if len(returns) == 0:
        return {
            "Annualized Return": 0,
            "Cumulative Return": 0,
            "Volatility": 0,
            "Sharpe Ratio": None,
            "Max Drawdown": 0,
            "Correlation with Portfolio 1": "-"
        }

    weighted_returns = (returns * weights).sum(axis=1)
    cumulative_return = (1 + weighted_returns).prod() - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(weighted_returns)) - 1
    volatility = weighted_returns.std() * np.sqrt(252)
    sharpe = annualized_return / volatility if volatility != 0 else np.nan

    cumulative = (1 + weighted_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    correlation = weighted_returns.corr(reference_returns) if reference_returns is not None else np.nan
    cumulative_perf = (1 + weighted_returns).prod() - 1

    return {
        "Annualized Return": round(annualized_return * 100, 2),
        "Cumulative Return": round(cumulative_perf * 100, 2),
        "Volatility": round(volatility * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown": round(max_drawdown * 100, 2),
        "Correlation with Portfolio 1": round(float(correlation), 2) if not np.isnan(correlation) else "-"

    }

# 🔢 Exécution pour les 3 portefeuilles
def analyze_all_portfolios(prices, portfolio_allocations):
    port_metrics = {}
    ref_returns = None

    crypto_label = f"Portefeuille 3 (60/40 + {crypto_global_pct:.0f}% Crypto)"
    name_mapping = {
        "Portfolio 1": "Portefeuille 1 (60/40)",
        "Portfolio 2": "Portefeuille 2 (60/40 + 5% Gold)",
        "Portfolio 3": crypto_label
    }

    for i, (name, alloc) in enumerate(portfolio_allocations.items()):
        metrics = compute_portfolio_metrics(prices, alloc, reference_returns=ref_returns)
        port_metrics[name_mapping.get(name, name)] = metrics

        if i == 0:
            tickers = [t for t in alloc if t in prices.columns]
            weights = np.array([alloc[t] for t in tickers])
            returns = prices[tickers].pct_change().dropna()
            ref_returns = (returns * weights).sum(axis=1)

    # ➕ Création & transposition du DataFrame
    df_metrics = pd.DataFrame(port_metrics).T
    numeric_cols = ["Annualized Return", "Cumulative Return", "Volatility", "Sharpe Ratio", "Max Drawdown", "Correlation with Portfolio 1"]

        # ✅ Convertir uniquement les colonnes numériques
    for col in numeric_cols:
         if col in df_metrics.columns:
            df_metrics[col] = pd.to_numeric(df_metrics[col], errors="coerce")

    # ➕ Transposer et arrondir les valeurs numériques à 2 décimales
    df_metrics_display = df_metrics.T
    df_metrics_display = df_metrics_display.applymap(
        lambda x: round(x, 2) if isinstance(x, (int, float)) else x
    )

    # 👉 Affichage brut sans style
    st.markdown("### Comparaison de portefeuilles")
    st.dataframe(df_metrics_display, use_container_width=True)

    return df_metrics_display
    
# Sélection utilisateur
available_assets = list(full_asset_mapping.keys())
selected_asset = st.selectbox("📌 Sélectionnez un actif :", available_assets)
asset_ticker = full_asset_mapping[selected_asset]

# Périodes disponibles
timeframes = {
    "1 semaine": "7d", "1 mois": "30d", "3 mois": "90d", "6 mois": "180d",
    "1 an": "365d","2 ans" : "730d","3 ans": "1095d", "5 ans": "1825d"
}
period_label = st.selectbox("⏳ Période :", list(timeframes.keys()))

# 🎯 Période personnalisée (optionnelle)
use_custom_period = st.checkbox("Période personnalisée")
custom_col1, custom_col2 = st.columns(2)
with custom_col1:
    custom_start = st.date_input("Date de début", value=pd.Timestamp.today() - pd.Timedelta(days=30), disabled=not use_custom_period)
with custom_col2:
    custom_end = st.date_input("Date de fin", value=pd.Timestamp.today() - pd.Timedelta(days=1), disabled=not use_custom_period)

# Actifs de comparaison (réorganisés par thème)
st.markdown("**Liste des actifs à comparer**")

compare_assets = [a for a in available_assets if a != selected_asset]
selected_comparisons = st.multiselect(
    "📊 Actifs à comparer :",
    compare_assets,
    default=[a for a in [
        "Bitcoin (BTC)", "Ethereum (ETH)",
        "MSCI World", "Nasdaq", "S&P 500",
        "US 10Y Yield", "Dollar Index", "Gold"
    ] if a != selected_asset]
)
compare_tickers = [full_asset_mapping[a] for a in selected_comparisons]

# ▶️ Bouton d’analyse
if st.button("🔎 Analyser"):
    try:
        tickers_graphiques = list(set(compare_tickers + [asset_ticker]))

        tickers_portefeuilles = set()
        for alloc in portfolio_allocations.values():
            tickers_portefeuilles.update(alloc.keys())
        tickers_dl = list(set(tickers_graphiques + list(tickers_portefeuilles)))

        # 📅 Détermination période finale : soit personnalisée soit prédéfinie
        if use_custom_period:
            start_date = pd.to_datetime(custom_start)
            end_date = pd.to_datetime(custom_end)
            period = None
        else:
            nb_days = int(timeframes[period_label].replace('d', ''))
            end_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize()
            start_date = end_date - pd.Timedelta(days=nb_days - 1)
            period = timeframes[period_label]

        all_days = pd.date_range(start=start_date, end=end_date, freq="D")
        raw_data = yf.download(tickers_dl, start=start_date, end=end_date + pd.Timedelta(days=1), interval="1d")["Close"]
        df = pd.DataFrame(index=all_days)
        missing_cols = [col for col in tickers_dl if col not in raw_data.columns]
        for col in missing_cols:
            raw_data[col] = pd.NA
        df = df.join(raw_data[tickers_dl])
        df_graph = df[tickers_graphiques]

        traditional_tickers = [full_asset_mapping[a] for a in asset_mapping.keys() if full_asset_mapping[a] in df.columns]
        df[traditional_tickers] = df[traditional_tickers].ffill()
        df = df.dropna(how="all")
        # Remplissage uniquement pour les actifs traditionnels
        df[traditional_tickers] = df[traditional_tickers].ffill().bfill()

        # Remplacement du label de période
        if not use_custom_period:
            label_period = f"sur {period_label.lower()}"
        else:
             label_period = f"du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}"

        # Matrice de corrélation
        returns = df_graph.pct_change().dropna()
        correlation_matrix = returns.corr()
        correlation_matrix = correlation_matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")

        asset_names = {v: k for k, v in full_asset_mapping.items()}
        correlation_matrix.rename(index=asset_names, columns=asset_names, inplace=True)
        

        fig_width = max(4, len(correlation_matrix.columns) * 0.3)
        fig, ax = plt.subplots(figsize=(fig_width, fig_width))
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom", ["#4E26DF", "#a993fa", "#CAE5F5", "#F2F2F2", "#C3F793","#7CEF17"], N=256
        )       

        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap=custom_cmap,
            cbar=True,
            cbar_kws={
              'shrink': 0.4,
             'format': '%.2f'
            },
            ax=ax,
            annot_kws={"fontsize": 5, "color": "#35434B"},
            linewidths=1,
            linecolor="white"
        )   

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=5, color="#35434B")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=30, va='top', fontsize=5, color="#35434B")
        ax.set_title(f"Matrice de corrélation {label_period}", fontsize=7, color="#35434B", pad=10)
        fig.tight_layout(pad=2.0)
        st.pyplot(fig)

        # Graphique de performances
        df = df[df.columns[df.notna().any()]].ffill().bfill()
        df_graph = df_graph.ffill().bfill()
        performance = df_graph.iloc[-1] / df_graph.iloc[0] - 1
        performance = performance.sort_values(ascending=False)
        perf_pct = (df_graph / df_graph.iloc[0] - 1) * 100
        perf_df = performance.reset_index()
        perf_df.columns = ['Actif', 'Performance (%)']
        perf_df['Performance (%)'] = perf_df['Performance (%)'].apply(lambda x: round(x * 100, 2))
        perf_df["Actif"] = perf_df["Actif"].map(asset_names_map).fillna(perf_df["Actif"])

        custom_colors = ["#7CEF17", "#4E26DF", "#35434B", "#949494", "#EDEDED", "#C3F793", "#B8A8F2", "#C1E5F5", "#F2F2F2", "#F2CFEE"]
        palette = custom_colors[:len(perf_df)] if len(perf_df) <= len(custom_colors) else custom_colors + sns.color_palette("husl", n_colors=len(perf_df) - len(custom_colors))

        fig_perf, ax_perf = plt.subplots(figsize=(5, 4))
        bars = ax_perf.bar(perf_df["Actif"], perf_df["Performance (%)"], color=palette)

        for bar in bars:
            height = bar.get_height()
            ax_perf.text(bar.get_x() + bar.get_width() / 2, height + (0.2 if height >= 0 else -0.4), f'{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=7, color='#35434B')

        ax_perf.axhline(0, color='#949494', linewidth=1, linestyle='--')
        y_range = max(abs(perf_df['Performance (%)'].max()), abs(perf_df['Performance (%)'].min())) * 1.2
        ax_perf.set_ylim(-y_range, y_range)
        ax_perf.set_xticklabels(ax_perf.get_xticklabels(), rotation=30, fontsize=6, color='#35434B')
        ax_perf.tick_params(axis='y', labelsize=6, labelcolor='#35434B')
        ax_perf.set_title(f"Performances cumulées {label_period}", fontsize=9, color="#35434B", pad=10)
        fig_perf.tight_layout(pad=2.0)
        st.pyplot(fig_perf)

        # 📈 Graphique des prix normalisés
        df_graph = df_graph.ffill().bfill()
        df_normalized = df_graph / df_graph.iloc[0] * 100
        fig_price, ax_price = plt.subplots(figsize=(6, 4))

        for idx, col in enumerate(df_normalized.columns):
            color = performance_colors[idx % len(performance_colors)]
            ax_price.plot(df_normalized.index, df_normalized[col], label=asset_names_map.get(col, col), color=color, linewidth=1.5)

        ax_price.set_title(f"Évolution de la performance des actifs {label_period}", fontsize=11, color="#35434B", pad=10)
        ax_price.legend(fontsize=6)
        ax_price.tick_params(axis='x', labelsize=6, labelcolor="#35434B")
        ax_price.tick_params(axis='y', labelsize=6, labelcolor="#35434B")
        ax_price.grid(True, linestyle='--', alpha=0.4)
        fig_price.tight_layout(pad=2.0)
        st.pyplot(fig_price)

        # 🔁 Calcul de la volatilité annualisée
        vol = df_graph.pct_change().std() * (252 ** 0.5) * 100
        vol_df = vol.sort_values(ascending=False).reset_index()
        vol_df.columns = ["Actif", "Volatilité (%)"]
        vol_df["Volatilité (%)"] = vol_df["Volatilité (%)"].apply(lambda x: round(x, 2))
        vol_df["Actif"] = vol_df["Actif"].map(asset_names_map).fillna(vol_df["Actif"])

        fig_vol, ax_vol = plt.subplots(figsize=(5, 4))
        bars_vol = ax_vol.bar(vol_df["Actif"], vol_df["Volatilité (%)"], color=palette[:len(vol_df)])

        for bar in bars_vol:
            height = bar.get_height()
            ax_vol.text(bar.get_x() + bar.get_width() / 2, height + 0.2, f'{height:.2f}%', ha='center', va='bottom', fontsize=7, color='#35434B')

        ax_vol.set_xticklabels(ax_vol.get_xticklabels(), rotation=30, fontsize=6, color='#35434B')
        ax_vol.tick_params(axis='y', labelsize=6, labelcolor='#35434B')
        ax_vol.set_title(f"Volatilité annualisée {label_period}", fontsize=9, color="#35434B", pad=10)
        fig_vol.tight_layout(pad=2.0)
        st.pyplot(fig_vol)

        # Analyse des portefeuilles
        
        if "Portfolio 3" not in portfolio_allocations and total_pct == 100 and crypto_global_pct > 0:
            portfolio_allocations["Portfolio 3"] = portfolio3
        df_portfolios = analyze_all_portfolios(df, portfolio_allocations)
        df.rename(columns=asset_names_map, inplace=True)

        # 📥 Téléchargements Excel & PDF
        st.subheader("📥 Exporter les résultats")
        
        # Excel multi-feuilles
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name="Prix")
            perf_pct.to_excel(writer, sheet_name="Performance (%)")
            vol.to_frame("Volatilité (%)").to_excel(writer, sheet_name="Volatilité (%)")
            df_portfolios.to_excel(writer, sheet_name="Résumé Portefeuilles")
        excel_buffer.seek(0)
        st.download_button("📄 Télécharger les données complètes (.xlsx)", data=excel_buffer, file_name="donnees_completes.xlsx")

        # PDF multi-graph
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            for fig_to_save in [fig, fig_perf, fig_price, fig_vol]:
                fig_to_save.tight_layout()
                pdf.savefig(fig_to_save)
                plt.close(fig_to_save)  # Très important !

        pdf_buffer.seek(0)
        st.download_button(
            "🖼️ Télécharger les graphiques en PDF",
            data=pdf_buffer,
            file_name="graphique_actifs.pdf",
            mime="application/pdf"
        )
    
    except Exception as e:
        st.error("❌ Erreur lors du chargement ou de l’analyse des données.")
        st.code(str(e))
        st.info("💡 Réessayez avec une période ou un actif différent.")