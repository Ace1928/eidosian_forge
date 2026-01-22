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
def GetTimeZone(name):
    """Returns a datetime.tzinfo object for name.

  Args:
    name: A timezone name string, None for the local timezone.

  Returns:
    A datetime.tzinfo object for name, local timezone if name is unknown.
  """
    if name in ('UTC', 'Z'):
        return UTC
    if name in ('LOCAL', 'L'):
        return LOCAL
    name = times_data.ABBREVIATION_TO_IANA.get(name, name)
    tzinfo = tz.gettz(name)
    if not tzinfo and tzwin:
        name = times_data.IANA_TO_WINDOWS.get(name, name)
        try:
            tzinfo = tzwin.tzwin(name)
        except WindowsError:
            pass
    return tzinfo