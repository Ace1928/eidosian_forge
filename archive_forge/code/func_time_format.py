import calendar
from datetime import date, datetime, time
from email.utils import format_datetime as format_datetime_rfc5322
from django.utils.dates import (
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import (
from django.utils.translation import gettext as _
def time_format(value, format_string):
    """Convenience function"""
    tf = TimeFormat(value)
    return tf.format(format_string)