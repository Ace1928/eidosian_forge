import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _yscale_validator(value):
    if isinstance(value, str) and value in ('linear', 'log', 'symlog', 'logit'):
        return True
    if not isinstance(value, dict):
        return False
    if not 'yscale' in value:
        return False
    yscale = value['yscale']
    if not (isinstance(yscale, str) and yscale in ('linear', 'log', 'symlog', 'logit')):
        return False
    return True