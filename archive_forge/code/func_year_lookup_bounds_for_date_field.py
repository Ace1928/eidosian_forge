import datetime
import decimal
import json
import warnings
from importlib import import_module
import sqlparse
from django.conf import settings
from django.db import NotSupportedError, transaction
from django.db.backends import utils
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.encoding import force_str
def year_lookup_bounds_for_date_field(self, value, iso_year=False):
    """
        Return a two-elements list with the lower and upper bound to be used
        with a BETWEEN operator to query a DateField value using a year
        lookup.

        `value` is an int, containing the looked-up year.
        If `iso_year` is True, return bounds for ISO-8601 week-numbering years.
        """
    if iso_year:
        first = datetime.date.fromisocalendar(value, 1, 1)
        second = datetime.date.fromisocalendar(value + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        first = datetime.date(value, 1, 1)
        second = datetime.date(value, 12, 31)
    first = self.adapt_datefield_value(first)
    second = self.adapt_datefield_value(second)
    return [first, second]