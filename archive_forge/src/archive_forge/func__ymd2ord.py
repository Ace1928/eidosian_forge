import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
@classmethod
def _ymd2ord(cls, year, month, day):
    return _post_epoch_days_before_year(year) + cls._DAYS_BEFORE_MONTH[month] + (month > 2 and calendar.isleap(year)) + day