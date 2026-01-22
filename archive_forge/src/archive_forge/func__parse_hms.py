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
def _parse_hms(self, idx, tokens, info, hms_idx):
    if hms_idx is None:
        hms = None
        new_idx = idx
    elif hms_idx > idx:
        hms = info.hms(tokens[hms_idx])
        new_idx = hms_idx
    else:
        hms = info.hms(tokens[hms_idx]) + 1
        new_idx = idx
    return (new_idx, hms)