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
def _updownhollow_colors(upcolor, downcolor, hollowcolor, opens, closes):
    if upcolor == downcolor:
        return upcolor
    umap = {True: hollowcolor, False: upcolor}
    dmap = {True: hollowcolor, False: downcolor}
    first = umap[closes[0] > opens[0]]
    _list = [umap[cls > opn] if cls > cls0 else dmap[cls > opn] for cls0, opn, cls in zip(closes[0:-1], opens[1:], closes[1:])]
    return [first] + _list