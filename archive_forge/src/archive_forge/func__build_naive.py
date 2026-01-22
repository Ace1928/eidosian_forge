from __future__ import unicode_literals
import datetime
import re
import string
import time
import warnings
from calendar import monthrange
from io import StringIO
import six
from six import integer_types, text_type
from decimal import Decimal
from warnings import warn
from .. import relativedelta
from .. import tz
def _build_naive(self, res, default):
    repl = {}
    for attr in ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond'):
        value = getattr(res, attr)
        if value is not None:
            repl[attr] = value
    if 'day' not in repl:
        cyear = default.year if res.year is None else res.year
        cmonth = default.month if res.month is None else res.month
        cday = default.day if res.day is None else res.day
        if cday > monthrange(cyear, cmonth)[1]:
            repl['day'] = monthrange(cyear, cmonth)[1]
    naive = default.replace(**repl)
    if res.weekday is not None and (not res.day):
        naive = naive + relativedelta.relativedelta(weekday=res.weekday)
    return naive