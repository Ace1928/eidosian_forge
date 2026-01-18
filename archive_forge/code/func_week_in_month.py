import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \
def week_in_month(dt: datetime.datetime) -> int:
    month_cal = calendar.monthcalendar(dt.year, dt.month)
    for k, week_cal in enumerate(month_cal, start=1):
        if dt.day in week_cal:
            if month_cal[0][3]:
                return k
            elif k > 1:
                return k - 1
            if dt.month > 1:
                prev_month_cal = calendar.monthcalendar(dt.year, dt.month - 1)
            else:
                prev_month_cal = calendar.monthcalendar(dt.year - 1, 12)
            if prev_month_cal[0][3]:
                return len(prev_month_cal)
            else:
                return len(prev_month_cal) - 1
    else:
        raise ValueError(f'{dt.day} does not match related calendar')