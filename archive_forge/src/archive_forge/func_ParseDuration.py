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
def ParseDuration(string, calendar=False, default_suffix=None):
    """Parses a duration string and returns a Duration object.

  Durations using only hours, miniutes, seconds and microseconds are exact.
  calendar=True allows the constructor to use duration units larger than hours.
  These durations will be inexact across daylight savings time and leap year
  boundaries, but will be "calendar" correct. For example:

    2015-02-14 + P1Y   => 2016-02-14
    2015-02-14 + P365D => 2016-02-14
    2016-02-14 + P1Y   => 2017-02-14
    2016-02-14 + P366D => 2017-02-14

    2016-03-13T01:00:00 + P1D   => 2016-03-14T01:00:00
    2016-03-13T01:00:00 + PT23H => 2016-03-14T01:00:00
    2016-03-13T01:00:00 + PT24H => 2016-03-14T03:00:00

  Args:
    string: The ISO 8601 duration/period string to parse.
    calendar: Use duration units larger than hours if True.
    default_suffix: Use this suffix if string is an unqualified int.

  Raises:
    DurationSyntaxError: Invalid duration syntax.
    DurationValueError: A Duration numeric constant exceeded its range.

  Returns:
    An iso_duration.Duration object for the given ISO 8601 duration/period
    string.
  """
    if default_suffix:
        try:
            seconds = int(string)
            string = '{}{}'.format(seconds, default_suffix)
        except ValueError:
            pass
    try:
        return iso_duration.Duration(calendar=calendar).Parse(string)
    except (AttributeError, OverflowError) as e:
        raise DurationValueError(six.text_type(e))
    except ValueError as e:
        raise DurationSyntaxError(six.text_type(e))