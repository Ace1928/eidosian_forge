import sys
import datetime
import locale as _locale
from itertools import repeat
def _monthlen(year, month):
    return mdays[month] + (month == February and isleap(year))