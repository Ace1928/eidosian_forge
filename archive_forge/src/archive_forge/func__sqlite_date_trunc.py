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
def _sqlite_date_trunc(lookup_type, dt, tzname, conn_tzname):
    dt = _sqlite_datetime_parse(dt, tzname, conn_tzname)
    if dt is None:
        return None
    if lookup_type == 'year':
        return f'{dt.year:04d}-01-01'
    elif lookup_type == 'quarter':
        month_in_quarter = dt.month - (dt.month - 1) % 3
        return f'{dt.year:04d}-{month_in_quarter:02d}-01'
    elif lookup_type == 'month':
        return f'{dt.year:04d}-{dt.month:02d}-01'
    elif lookup_type == 'week':
        dt -= timedelta(days=dt.weekday())
        return f'{dt.year:04d}-{dt.month:02d}-{dt.day:02d}'
    elif lookup_type == 'day':
        return f'{dt.year:04d}-{dt.month:02d}-{dt.day:02d}'
    raise ValueError(f'Unsupported lookup type: {lookup_type!r}')