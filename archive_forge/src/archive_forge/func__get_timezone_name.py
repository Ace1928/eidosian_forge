import functools
import zoneinfo
from contextlib import ContextDecorator
from datetime import datetime, timedelta, timezone, tzinfo
from asgiref.local import Local
from django.conf import settings
def _get_timezone_name(timezone):
    """
    Return the offset for fixed offset timezones, or the name of timezone if
    not set.
    """
    return timezone.tzname(None) or str(timezone)