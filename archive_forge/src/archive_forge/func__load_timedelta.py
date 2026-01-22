import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
@functools.lru_cache(maxsize=512)
def _load_timedelta(seconds):
    return timedelta(seconds=seconds)