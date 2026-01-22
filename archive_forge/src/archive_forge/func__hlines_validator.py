import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _hlines_validator(value):
    if isinstance(value, dict):
        if 'hlines' in value:
            value = value['hlines']
        else:
            return False
    return isinstance(value, (float, int)) or (isinstance(value, (list, tuple)) and all([isinstance(v, (float, int)) for v in value]))