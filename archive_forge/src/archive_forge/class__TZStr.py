import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
class _TZStr:
    __slots__ = ('std', 'dst', 'start', 'end', 'get_trans_info', 'get_trans_info_fromutc', 'dst_diff')

    def __init__(self, std_abbr, std_offset, dst_abbr, dst_offset, start=None, end=None):
        self.dst_diff = dst_offset - std_offset
        std_offset = _load_timedelta(std_offset)
        self.std = _ttinfo(utcoff=std_offset, dstoff=_load_timedelta(0), tzname=std_abbr)
        self.start = start
        self.end = end
        dst_offset = _load_timedelta(dst_offset)
        delta = _load_timedelta(self.dst_diff)
        self.dst = _ttinfo(utcoff=dst_offset, dstoff=delta, tzname=dst_abbr)
        assert start is not None, 'No transition start specified'
        assert end is not None, 'No transition end specified'
        self.get_trans_info = self._get_trans_info
        self.get_trans_info_fromutc = self._get_trans_info_fromutc

    def transitions(self, year):
        start = self.start.year_to_epoch(year)
        end = self.end.year_to_epoch(year)
        return (start, end)

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