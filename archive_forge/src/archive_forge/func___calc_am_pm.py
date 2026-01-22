import time
import locale
import calendar
from re import compile as re_compile
from re import IGNORECASE
from re import escape as re_escape
from datetime import (date as datetime_date,
from _thread import allocate_lock as _thread_allocate_lock
def __calc_am_pm(self):
    am_pm = []
    for hour in (1, 22):
        time_tuple = time.struct_time((1999, 3, 17, hour, 44, 55, 2, 76, 0))
        am_pm.append(time.strftime('%p', time_tuple).lower())
    self.am_pm = am_pm