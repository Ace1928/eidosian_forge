import sys
import datetime
import locale as _locale
from itertools import repeat
def _prevmonth(year, month):
    if month == 1:
        return (year - 1, 12)
    else:
        return (year, month - 1)