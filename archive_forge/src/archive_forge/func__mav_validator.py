import matplotlib.dates  as mdates
import pandas   as pd
import numpy    as np
import datetime
from   mplfinance._helpers import _list_of_dict, _mpf_is_color_like
from   mplfinance._helpers import _num_or_seq_of_num
import matplotlib as mpl
import warnings
def _mav_validator(mav_value):
    """
    Value for mav (moving average) keyword may be:
    scalar int greater than 1, or tuple of ints, or list of ints (each greater than 1)
    or a dict of `period` and `shift` each of which may be:
    scalar int, or tuple of ints, or list of ints: each `period` int must be greater than 1
    """

    def _valid_mav(value, is_period=True):
        if not isinstance(value, (tuple, list, int)):
            return False
        if isinstance(value, int):
            return value >= 2 or not is_period
        for num in value:
            if not isinstance(num, int) or (is_period and num < 2):
                return False
        return True
    if not isinstance(mav_value, (tuple, list, int, dict)):
        return False
    if not isinstance(mav_value, dict):
        return _valid_mav(mav_value)
    else:
        if 'period' not in mav_value:
            return False
        period = mav_value['period']
        if not _valid_mav(period):
            return False
        if 'shift' not in mav_value:
            return True
        shift = mav_value['shift']
        if not _valid_mav(shift, False):
            return False
        if isinstance(period, int) and isinstance(shift, int):
            return True
        if isinstance(period, (tuple, list)) and isinstance(shift, (tuple, list)):
            if len(period) != len(shift):
                return False
            return True
        return False