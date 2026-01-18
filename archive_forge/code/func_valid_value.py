import copy
import datetime
import json
import math
import operator
import os
import re
import uuid
import warnings
from decimal import Decimal, DecimalException
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit
from django.conf import settings
from django.core import validators
from django.core.exceptions import ValidationError
from django.forms.boundfield import BoundField
from django.forms.utils import from_current_timezone, to_current_timezone
from django.forms.widgets import (
from django.utils import formats
from django.utils.choices import normalize_choices
from django.utils.dateparse import parse_datetime, parse_duration
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.duration import duration_string
from django.utils.ipv6 import clean_ipv6_address
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
def valid_value(self, value):
    """Check to see if the provided value is a valid choice."""
    text_value = str(value)
    for k, v in self.choices:
        if isinstance(v, (list, tuple)):
            for k2, v2 in v:
                if value == k2 or text_value == str(k2):
                    return True
        elif value == k or text_value == str(k):
            return True
    return False