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
def _date_to_iloc_extrapolate(dtseries, date):
    """Convert a `date` to a location, given a date series w/a datetime index.
       If `date` does not exactly match a date in the series then interpolate between two dates.
       If `date` is outside the range of dates in the series, then extrapolate:
       Extrapolation results in increased error as the distance of the extrapolation increases.
       We have two methods to extrapolate:
       (1) Determine a linear equation based on the data provided in `dtseries`,
           and use that equation to calculate the location for the date.
       (2) Multiply by 5/7 the number of days between the edge date of dtseries and the
           date for which we are requesting a location.
           THIS ASSUMES DAILY data AND a 5 DAY TRADING WEEK.
       Empirical observation (scratch_pad/date_to_iloc_extrapolation.ipynb) shows that
       the systematic error of these two methods tends to be in opposite directions:
       taking the average of the two methods reduces systematic errorr: However,
       since method (2) applies only to DAILY data, we take the average of the two
       methods only for daily data.  For intraday data we use only method (1).
    """
    d1s = dtseries.loc[date:]
    if len(d1s) < 1:
        loc_linear = _date_to_iloc_linear(dtseries, date)
        loc_5_7ths = _date_to_iloc_5_7ths(dtseries, date, 'forward')
        if loc_5_7ths is not None:
            return (loc_linear + loc_5_7ths) / 2.0
        else:
            return loc_linear
    d1 = d1s.index[0]
    d2s = dtseries.loc[:date]
    if len(d2s) < 1:
        loc_linear = _date_to_iloc_linear(dtseries, date)
        loc_5_7ths = _date_to_iloc_5_7ths(dtseries, date, 'backward')
        if loc_5_7ths is not None:
            return (loc_linear + loc_5_7ths) / 2.0
        else:
            return loc_linear
    d2 = dtseries.loc[:date].index[-1]
    loc1 = dtseries.index.get_loc(d1)
    if isinstance(loc1, slice):
        loc1 = loc1.start
    loc2 = dtseries.index.get_loc(d2)
    if isinstance(loc2, slice):
        loc2 = loc2.stop - 1
    return (loc1 + loc2) / 2.0