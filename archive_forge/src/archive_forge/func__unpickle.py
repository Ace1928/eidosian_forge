import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
@classmethod
def _unpickle(cls, key, from_cache, /):
    if from_cache:
        return cls(key)
    else:
        return cls.no_cache(key)