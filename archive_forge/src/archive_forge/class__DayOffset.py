import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
class _DayOffset:
    __slots__ = ['d', 'julian', 'hour', 'minute', 'second']

    def __init__(self, d, julian, hour=2, minute=0, second=0):
        min_day = 0 + julian
        if not min_day <= d <= 365:
            raise ValueError(f'd must be in [{min_day}, 365], not: {d}')
        self.d = d
        self.julian = julian
        self.hour = hour
        self.minute = minute
        self.second = second

    def year_to_epoch(self, year):
        days_before_year = _post_epoch_days_before_year(year)
        d = self.d
        if self.julian and d >= 59 and calendar.isleap(year):
            d += 1
        epoch = (days_before_year + d) * 86400
        epoch += self.hour * 3600 + self.minute * 60 + self.second
        return epoch