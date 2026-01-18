import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
def year_to_epoch(self, year):
    """Calculates the datetime of the occurrence from the year"""
    first_day, days_in_month = calendar.monthrange(year, self.m)
    month_day = (self.d - (first_day + 1)) % 7 + 1
    month_day += (self.w - 1) * 7
    if month_day > days_in_month:
        month_day -= 7
    ordinal = self._ymd2ord(year, self.m, month_day)
    epoch = ordinal * 86400
    epoch += self.hour * 3600 + self.minute * 60 + self.second
    return epoch