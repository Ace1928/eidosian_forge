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
def _could_be_tzname(self, hour, tzname, tzoffset, token):
    return hour is not None and tzname is None and (tzoffset is None) and (len(token) <= 5) and (all((x in string.ascii_uppercase for x in token)) or token in self.info.UTCZONE)