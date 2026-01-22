import time
import locale
import calendar
from re import compile as re_compile
from re import IGNORECASE
from re import escape as re_escape
from datetime import (date as datetime_date,
from _thread import allocate_lock as _thread_allocate_lock
def __calc_timezone(self):
    try:
        time.tzset()
    except AttributeError:
        pass
    self.tzname = time.tzname
    self.daylight = time.daylight
    no_saving = frozenset({'utc', 'gmt', self.tzname[0].lower()})
    if self.daylight:
        has_saving = frozenset({self.tzname[1].lower()})
    else:
        has_saving = frozenset()
    self.timezone = (no_saving, has_saving)