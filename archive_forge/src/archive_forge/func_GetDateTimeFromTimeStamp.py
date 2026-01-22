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
def GetDateTimeFromTimeStamp(timestamp, tzinfo=LOCAL):
    """Returns the datetime object for a UNIX timestamp.

  Args:
    timestamp: A UNIX timestamp in int or float seconds since the epoch
      (1970-01-01T00:00:00.000000Z).
    tzinfo: A tzinfo object for the timestamp timezone, None for naive.

  Raises:
    DateTimeValueError: A date/time numeric constant exceeds its range.

  Returns:
    The datetime object for a UNIX timestamp.
  """
    try:
        return datetime.datetime.fromtimestamp(timestamp, tzinfo)
    except (ValueError, OSError, OverflowError) as e:
        raise DateTimeValueError(six.text_type(e))