import sys
import datetime
import locale as _locale
from itertools import repeat
def weekday(year, month, day):
    """Return weekday (0-6 ~ Mon-Sun) for year, month (1-12), day (1-31)."""
    if not datetime.MINYEAR <= year <= datetime.MAXYEAR:
        year = 2000 + year % 400
    return datetime.date(year, month, day).weekday()