import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
@staticmethod
def _ts_to_local(trans_idx, trans_list_utc, utcoffsets):
    """Generate number of seconds since 1970 *in the local time*.

        This is necessary to easily find the transition times in local time"""
    if not trans_list_utc:
        return [[], []]
    trans_list_wall = [list(trans_list_utc), list(trans_list_utc)]
    if len(utcoffsets) > 1:
        offset_0 = utcoffsets[0]
        offset_1 = utcoffsets[trans_idx[0]]
        if offset_1 > offset_0:
            offset_1, offset_0 = (offset_0, offset_1)
    else:
        offset_0 = offset_1 = utcoffsets[0]
    trans_list_wall[0][0] += offset_0
    trans_list_wall[1][0] += offset_1
    for i in range(1, len(trans_idx)):
        offset_0 = utcoffsets[trans_idx[i - 1]]
        offset_1 = utcoffsets[trans_idx[i]]
        if offset_1 > offset_0:
            offset_1, offset_0 = (offset_0, offset_1)
        trans_list_wall[0][i] += offset_0
        trans_list_wall[1][i] += offset_1
    return trans_list_wall