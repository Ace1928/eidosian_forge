import functools
import random
import statistics
import zoneinfo
from datetime import timedelta
from hashlib import md5, sha1, sha224, sha256, sha384, sha512
from math import (
from re import search as re_search
from django.db.backends.utils import (
from django.utils import timezone
from django.utils.duration import duration_microseconds
def _sqlite_regexp(pattern, string):
    if pattern is None or string is None:
        return None
    if not isinstance(string, str):
        string = str(string)
    return bool(re_search(pattern, string))