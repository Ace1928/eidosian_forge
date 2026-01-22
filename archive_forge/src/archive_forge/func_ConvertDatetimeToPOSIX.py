from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from calendar import timegm
import getpass
import logging
import os
import re
import time
import six
from gslib.exception import CommandException
from gslib.tz_utc import UTC
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.metadata_util import GetValueFromObjectCustomMetadata
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.unit_util import SECONDS_PER_DAY
def ConvertDatetimeToPOSIX(dt):
    """Converts a datetime object to UTC and formats as POSIX.

  Sanitize the timestamp returned in dt, and put it in UTC format. For more
  information see the UTC class.

  Args:
    dt: A Python datetime object.

  Returns:
    A POSIX timestamp according to UTC.
  """
    return long(timegm(dt.replace(tzinfo=UTC()).timetuple()))