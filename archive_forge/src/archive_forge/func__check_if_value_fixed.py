import copy
import datetime
import decimal
import operator
import uuid
import warnings
from base64 import b64decode, b64encode
from functools import partialmethod, total_ordering
from django import forms
from django.apps import apps
from django.conf import settings
from django.core import checks, exceptions, validators
from django.db import connection, connections, router
from django.db.models.constants import LOOKUP_SEP
from django.db.models.query_utils import DeferredAttribute, RegisterLookupMixin
from django.utils import timezone
from django.utils.choices import (
from django.utils.datastructures import DictWrapper
from django.utils.dateparse import (
from django.utils.duration import duration_microseconds, duration_string
from django.utils.functional import Promise, cached_property
from django.utils.ipv6 import clean_ipv6_address
from django.utils.itercompat import is_iterable
from django.utils.text import capfirst
from django.utils.translation import gettext_lazy as _
def _check_if_value_fixed(self, value, now=None):
    """
        Check if the given value appears to have been provided as a "fixed"
        time value, and include a warning in the returned list if it does. The
        value argument must be a date object or aware/naive datetime object. If
        now is provided, it must be a naive datetime object.
        """
    if now is None:
        now = _get_naive_now()
    offset = datetime.timedelta(seconds=10)
    lower = now - offset
    upper = now + offset
    if isinstance(value, datetime.datetime):
        value = _to_naive(value)
    else:
        assert isinstance(value, datetime.date)
        lower = lower.date()
        upper = upper.date()
    if lower <= value <= upper:
        return [checks.Warning('Fixed default value provided.', hint='It seems you set a fixed date / time / datetime value as default for this field. This may not be what you want. If you want to have the current date as default, use `django.utils.timezone.now`', obj=self, id='fields.W161')]
    return []