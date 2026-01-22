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
def _construct_renko_collections(dates, highs, lows, volumes, config_renko_params, closes, marketcolors=None):
    """Represent the price change with bricks

    NOTE: this code assumes if any value open, low, high, close is
    missing they all are missing

    Algorithm Explanation
    ---------------------
    In the first part of the algorithm, we populate the cdiff array
    along with adjusting the dates and volumes arrays into the new_dates and
    new_volumes arrays. A single date includes a range from no bricks to many
    bricks, if a date has no bricks it shall not be included in new_dates,
    and if it has n bricks then it will be included n times. Volumes use a
    volume cache to save volume amounts for dates that do not have any bricks
    before adding the cache to the next date that has at least one brick.
    We populate the cdiff array with each close values difference from the
    previously created brick divided by the brick size.

    In the second part of the algorithm, we iterate through the values in cdiff
    and add 1s or -1s to the bricks array depending on whether the value is
    positive or negative. Every time there is a trend change (ex. previous brick is
    an upbrick, current brick is a down brick) we draw one less brick to account
    for the price having to move the previous bricks amount before creating a
    brick in the opposite direction.

    In the final part of the algorithm, we enumerate through the bricks array and
    assign up-colors or down-colors to the associated index in the color array and
    populate the verts list with each bricks vertice to be used to create the matplotlib
    PolyCollection.

    Useful sources:
    https://avilpage.com/2018/01/how-to-plot-renko-charts-with-python.html
    https://school.stockcharts.com/doku.php?id=chart_analysis:renko

    Parameters
    ----------
    dates : sequence
        sequence of dates
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    config_renko_params : kwargs table (dictionary)
        brick_size : size of each brick
        atr_length : length of time used for calculating atr
    closes : sequence
        sequence of closing values
    marketcolors : dict of colors: up, down, edge, wick, alpha

    Returns
    -------
    ret : list
        rectCollection
    """
    renko_params = _process_kwargs(config_renko_params, _valid_renko_kwargs())
    if marketcolors is None:
        marketcolors = _get_mpfstyle('classic')['marketcolors']
    brick_size = renko_params['brick_size']
    atr_length = renko_params['atr_length']
    if brick_size == 'atr':
        if atr_length == 'total':
            brick_size = _calculate_atr(len(closes) - 1, highs, lows, closes)
        else:
            brick_size = _calculate_atr(atr_length, highs, lows, closes)
    else:
        upper_limit = (max(closes) - min(closes)) / 2
        lower_limit = 0.01 * _calculate_atr(len(closes) - 1, highs, lows, closes)
        if brick_size > upper_limit:
            raise ValueError('Specified brick_size may not be larger than (50% of the close price range of the dataset) which has value: ' + str(upper_limit))
        elif brick_size < lower_limit:
            raise ValueError('Specified brick_size may not be smaller than (0.01* the Average True Value of the dataset) which has value: ' + str(lower_limit))
    alpha = marketcolors['alpha']
    uc = mcolors.to_rgba(marketcolors['candle']['up'], alpha)
    dc = mcolors.to_rgba(marketcolors['candle']['down'], alpha)
    euc = mcolors.to_rgba(marketcolors['edge']['up'], 1.0)
    edc = mcolors.to_rgba(marketcolors['edge']['down'], 1.0)
    cdiff = []
    prev_close_brick = closes[0]
    volume_cache = 0
    new_dates = []
    new_volumes = []
    for i in range(len(closes) - 1):
        brick_diff = int((closes[i + 1] - prev_close_brick) / brick_size)
        if brick_diff == 0:
            if volumes is not None:
                volume_cache += volumes[i]
            continue
        cdiff.extend([int(brick_diff / abs(brick_diff))] * abs(brick_diff))
        if volumes is not None:
            new_volumes.extend([volumes[i] + volume_cache] * abs(brick_diff))
            volume_cache = 0
        new_dates.extend([dates[i]] * abs(brick_diff))
        prev_close_brick += brick_diff * brick_size
    bricks = []
    curr_price = closes[0]
    last_diff_sign = 0
    dates_volumes_index = 0
    for diff in cdiff:
        curr_diff_sign = diff / abs(diff)
        if last_diff_sign != 0 and curr_diff_sign != last_diff_sign:
            last_diff_sign = curr_diff_sign
            new_dates.pop(dates_volumes_index)
            if volumes is not None:
                if dates_volumes_index == len(new_volumes) - 1:
                    new_volumes[dates_volumes_index - 1] += new_volumes[dates_volumes_index]
                else:
                    new_volumes[dates_volumes_index + 1] += new_volumes[dates_volumes_index]
                new_volumes.pop(dates_volumes_index)
            continue
        last_diff_sign = curr_diff_sign
        if diff > 0:
            bricks.extend([1] * abs(diff))
        else:
            bricks.extend([-1] * abs(diff))
        dates_volumes_index += 1
    verts = []
    colors = []
    edge_colors = []
    brick_values = []
    for index, number in enumerate(bricks):
        if number == 1:
            colors.append(uc)
            edge_colors.append(euc)
        else:
            colors.append(dc)
            edge_colors.append(edc)
        curr_price += brick_size * number
        brick_values.append(curr_price)
        x, y = (index, curr_price)
        verts.append(((x, y), (x, y + brick_size), (x + 1, y + brick_size), (x + 1, y)))
    useAA = (0,)
    lw = None
    rectCollection = PolyCollection(verts, facecolors=colors, antialiaseds=useAA, edgecolors=edge_colors, linewidths=lw)
    calculated_values = dict(dates=new_dates, volumes=new_volumes, values=brick_values, size=brick_size)
    return ([rectCollection], calculated_values)