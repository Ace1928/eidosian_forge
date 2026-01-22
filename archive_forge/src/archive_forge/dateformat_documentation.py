import calendar
from datetime import date, datetime, time
from email.utils import format_datetime as format_datetime_rfc5322
from django.utils.dates import (
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import (
from django.utils.translation import gettext as _
Day of the year, i.e. 1 to 366.