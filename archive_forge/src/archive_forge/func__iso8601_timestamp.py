import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
def _iso8601_timestamp(timestamp):
    """Produces an ISO8601 datetime.

  Args:
    timestamp: an Epoch based timestamp in seconds.

  Returns:
    A iso8601 format timestamp if the input is a valid timestamp, None otherwise
  """
    if timestamp is None or timestamp < 0:
        return None
    return datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc).isoformat()