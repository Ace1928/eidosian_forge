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
def _tline_lsq(dfslice, tline_use):
    """
        This closed-form linear least squares algorithm was taken from:
        https://mmas.github.io/least-squares-fitting-numpy-scipy
        """
    si = dfslice[tline_use].mean(axis=1)
    s = si.dropna()
    if len(s) < 2:
        err = 'NOT enough data for Least Squares'
        if len(si) > 2:
            err += ', due to presence of NaNs'
        raise ValueError(err)
    xs = mdates.date2num(s.index.to_pydatetime())
    ys = s.values
    a = np.vstack([xs, np.ones(len(xs))]).T
    m, b = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, ys))
    x1, x2 = (xs[0], xs[-1])
    y1 = m * x1 + b
    y2 = m * x2 + b
    x1, x2 = (mdates.num2date(x1).replace(tzinfo=None), mdates.num2date(x2).replace(tzinfo=None))
    return ((x1, y1), (x2, y2))