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
def _date_to_mdate(date):
    if isinstance(date, str):
        pydt = pd.to_datetime(date).to_pydatetime()
    elif isinstance(date, pd.Timestamp):
        pydt = date.to_pydatetime()
    elif isinstance(date, (datetime.datetime, datetime.date)):
        pydt = date
    else:
        return None
    return mdates.date2num(pydt)