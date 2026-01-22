from datetime import datetime, timedelta, tzinfo
from bisect import bisect_right
import pytz
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError
class BaseTzInfo(tzinfo):
    _utcoffset = None
    _tzname = None
    zone = None

    def __str__(self):
        return self.zone