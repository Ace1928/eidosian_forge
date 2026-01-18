import os
import time
from datetime import datetime, timezone
import calendar

    Convert a tuple or struct_time representing a time as returned by gmtime()
    or localtime() to a 24-character string of the following form:

    >>> asctime(time.gmtime(0))
    'Thu Jan  1 00:00:00 1970'

    If t is not provided, the current time as returned by localtime() is used.
    Locale information is not used by asctime().

    This is meant to normalise the output of the built-in time.asctime() across
    different platforms and Python versions.
    In Python 3.x, the day of the month is right-justified, whereas on Windows
    Python 2.7 it is padded with zeros.

    See https://github.com/fonttools/fonttools/issues/455
    