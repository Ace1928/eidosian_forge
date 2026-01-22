from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import re
from dateutil import parser
from dateutil import tz
from dateutil.tz import _common as tz_common
import enum
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times_data
import six
def ParseDateTime(string, fmt=None, tzinfo=LOCAL):
    """Parses a date/time string and returns a datetime.datetime object.

  Args:
    string: The date/time string to parse. This can be a parser.parse()
      date/time or an ISO 8601 duration after Now(tzinfo) or before if prefixed
      by '-'.
    fmt: The input must satisfy this strptime(3) format string.
    tzinfo: A default timezone tzinfo object to use if string has no timezone.

  Raises:
    DateTimeSyntaxError: Invalid date/time/duration syntax.
    DateTimeValueError: A date/time numeric constant exceeds its range.

  Returns:
    A datetime.datetime object for the given date/time string.
  """
    if fmt:
        dt = _StrPtime(string, fmt)
        if tzinfo and (not dt.tzinfo):
            dt = dt.replace(tzinfo=tzinfo)
        return dt
    defaults = GetDateTimeDefaults(tzinfo=tzinfo)
    tzgetter = _TzInfoOrOffsetGetter()
    exc = None
    try:
        dt = parser.parse(string, tzinfos=tzgetter.Get, default=defaults)
        if tzinfo and (not tzgetter.timezone_was_specified):
            dt = parser.parse(string, tzinfos=None, default=defaults)
            dt = dt.replace(tzinfo=tzinfo)
        return dt
    except OverflowError as e:
        exc = exceptions.ExceptionContext(DateTimeValueError(six.text_type(e)))
    except (AttributeError, ValueError, TypeError) as e:
        exc = exceptions.ExceptionContext(DateTimeSyntaxError(six.text_type(e)))
        if not tzgetter.timezone_was_specified:
            prefix, explicit_tzinfo = _SplitTzFromDate(string)
            if explicit_tzinfo:
                try:
                    dt = parser.parse(prefix, default=defaults)
                except OverflowError as e:
                    exc = exceptions.ExceptionContext(DateTimeValueError(six.text_type(e)))
                except (AttributeError, ValueError, TypeError) as e:
                    exc = exceptions.ExceptionContext(DateTimeSyntaxError(six.text_type(e)))
                else:
                    return dt.replace(tzinfo=explicit_tzinfo)
    try:
        return ParseDuration(string).GetRelativeDateTime(Now(tzinfo=tzinfo))
    except Error:
        exc.Reraise()