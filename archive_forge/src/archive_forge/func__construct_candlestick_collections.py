import numpy  as np
import pandas as pd
import matplotlib.dates as mdates
import datetime
from itertools import cycle
from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.patches     import Ellipse
from matplotlib.collections import LineCollection, PolyCollection, PatchCollection
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from mplfinance._arg_validators import _alines_validator, _bypass_kwarg_validation
from mplfinance._arg_validators import _xlim_validator, _is_datelike
from mplfinance._styles         import _get_mpfstyle
from mplfinance._helpers        import _mpf_to_rgba
from six.moves import zip
from matplotlib.ticker import Formatter
def _construct_candlestick_collections(dates, opens, highs, lows, closes, marketcolors=None, config=None):
    """Represent the open, close as a bar line and high low range as a
    vertical line.

    NOTE: this code assumes if any value open, low, high, close is
    missing they all are missing


    Parameters
    ----------
    opens : sequence
        sequence of opening values
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    closes : sequence
        sequence of closing values
    marketcolors : dict of colors: up, down, edge, wick, alpha
    alpha : float
        bar transparency

    Returns
    -------
    ret : list
        (lineCollection, barCollection)
    """
    _check_input(opens, highs, lows, closes)
    if marketcolors is None:
        marketcolors = _get_mpfstyle('classic')['marketcolors']
    datalen = len(dates)
    avg_dist_between_points = (dates[-1] - dates[0]) / float(datalen)
    delta = config['_width_config']['candle_width'] / 2.0
    barVerts = [((date - delta, open), (date - delta, close), (date + delta, close), (date + delta, open)) for date, open, close in zip(dates, opens, closes)]
    rangeSegLow = [((date, low), (date, min(open, close))) for date, low, open, close in zip(dates, lows, opens, closes)]
    rangeSegHigh = [((date, high), (date, max(open, close))) for date, high, open, close in zip(dates, highs, opens, closes)]
    rangeSegments = rangeSegLow + rangeSegHigh
    alpha = marketcolors['alpha']
    overrides = config['marketcolor_overrides']
    faceonly = config['mco_faceonly']
    colors = _make_updown_color_list('candle', marketcolors, opens, closes, overrides)
    colors = [_mpf_to_rgba(c, alpha) for c in colors]
    if faceonly:
        overrides = None
    edgecolor = _make_updown_color_list('edge', marketcolors, opens, closes, overrides)
    wickcolor = _make_updown_color_list('wick', marketcolors, opens, closes, overrides)
    lw = config['_width_config']['candle_linewidth']
    rangeCollection = LineCollection(rangeSegments, colors=wickcolor, linewidths=lw)
    barCollection = PolyCollection(barVerts, facecolors=colors, edgecolors=edgecolor, linewidths=lw)
    return [rangeCollection, barCollection]