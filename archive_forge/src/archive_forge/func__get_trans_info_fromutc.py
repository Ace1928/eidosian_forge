import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
def _get_trans_info_fromutc(self, ts, year):
    start, end = self.transitions(year)
    start -= self.std.utcoff.total_seconds()
    end -= self.dst.utcoff.total_seconds()
    if start < end:
        isdst = start <= ts < end
    else:
        isdst = not end <= ts < start
    if self.dst_diff > 0:
        ambig_start = end
        ambig_end = end + self.dst_diff
    else:
        ambig_start = start
        ambig_end = start - self.dst_diff
    fold = ambig_start <= ts < ambig_end
    return (self.dst if isdst else self.std, fold)