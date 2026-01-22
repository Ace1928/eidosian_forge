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
def _parsems(self, value):
    """Parse a I[.F] seconds value into (seconds, microseconds)."""
    if '.' not in value:
        return (int(value), 0)
    else:
        i, f = value.split('.')
        return (int(i), int(f.ljust(6, '0')[:6]))