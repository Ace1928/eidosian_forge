import time as _time
import math as _math
import sys
from operator import index as _index
def _find_isoformat_datetime_separator(dtstr):
    len_dtstr = len(dtstr)
    if len_dtstr == 7:
        return 7
    assert len_dtstr > 7
    date_separator = '-'
    week_indicator = 'W'
    if dtstr[4] == date_separator:
        if dtstr[5] == week_indicator:
            if len_dtstr < 8:
                raise ValueError('Invalid ISO string')
            if len_dtstr > 8 and dtstr[8] == date_separator:
                if len_dtstr == 9:
                    raise ValueError('Invalid ISO string')
                if len_dtstr > 10 and _is_ascii_digit(dtstr[10]):
                    return 8
                return 10
            else:
                return 8
        else:
            return 10
    elif dtstr[4] == week_indicator:
        idx = 7
        while idx < len_dtstr:
            if not _is_ascii_digit(dtstr[idx]):
                break
            idx += 1
        if idx < 9:
            return idx
        if idx % 2 == 0:
            return 7
        else:
            return 8
    else:
        return 8