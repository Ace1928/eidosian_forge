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
def GetTimeStampFromDateTime(dt, tzinfo=LOCAL):
    """Returns the float UNIX timestamp (with microseconds) for dt.

  Args:
    dt: The datetime object to convert from.
    tzinfo: Use this tzinfo if dt is naiive.

  Returns:
    The float UNIX timestamp (with microseconds) for dt.
  """
    if not dt.tzinfo and tzinfo:
        dt = dt.replace(tzinfo=tzinfo)
    delta = dt - datetime.datetime.fromtimestamp(0, UTC)
    return delta.total_seconds()