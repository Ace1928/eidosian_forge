import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
def _get_trans_info(self, ts, year, fold):
    """Get the information about the current transition - tti"""
    start, end = self.transitions(year)
    if fold == (self.dst_diff >= 0):
        end -= self.dst_diff
    else:
        start += self.dst_diff
    if start < end:
        isdst = start <= ts < end
    else:
        isdst = not end <= ts < start
    return self.dst if isdst else self.std