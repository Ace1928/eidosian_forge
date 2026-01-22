import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _fill_between_validator(value):
    if _num_or_seq_of_num(value):
        return True
    if _valid_fb_dict(value):
        return True
    if _list_of_dict(value):
        return all([_valid_fb_dict(v) for v in value])
    return False