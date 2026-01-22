import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _xlim_validator(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and (all([isinstance(v, (int, float)) for v in value]) or all([_is_datelike(v) for v in value]))