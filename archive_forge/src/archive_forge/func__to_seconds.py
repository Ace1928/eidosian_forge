from datetime import datetime, timedelta, tzinfo
from bisect import bisect_right
import pytz
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError
def _to_seconds(td):
    """Convert a timedelta to seconds"""
    return td.seconds + td.days * 24 * 60 * 60