from mplfinance._styledata import default
from mplfinance._styledata import nightclouds
from mplfinance._styledata import classic
from mplfinance._styledata import mike
from mplfinance._styledata import charles
from mplfinance._styledata import blueskies
from mplfinance._styledata import starsandstripes
from mplfinance._styledata import sas
from mplfinance._styledata import brasil
from mplfinance._styledata import yahoo
from mplfinance._styledata import checkers
from mplfinance._styledata import binance
from mplfinance._styledata import kenan
from mplfinance._styledata import ibd
from mplfinance._styledata import binancedark
from mplfinance._styledata import tradingview
def _validate_style(style):
    keys = ['base_mpl_style', 'marketcolors', 'mavcolors', 'y_on_right', 'gridcolor', 'gridstyle', 'facecolor', 'rc']
    for key in keys:
        if key not in style.keys():
            err = f'Key "{key}" not found in style:\n\n    {style}'
            raise ValueError(err)
    mktckeys = ['candle', 'edge', 'wick', 'ohlc', 'volume', 'alpha']
    for key in mktckeys:
        if key not in style['marketcolors'].keys():
            err = f'Key "{key}" not found in marketcolors for style:\n\n    {style}'
            raise ValueError(err)
    if 'vcedge' not in style['marketcolors']:
        style['marketcolors']['vcedge'] = style['marketcolors']['volume']
    if 'vcdopcod' not in style['marketcolors']:
        style['marketcolors']['vcdopcod'] = False