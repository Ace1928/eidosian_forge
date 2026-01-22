from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StandardUnitsValueValuesEnum(_messages.Enum):
    """System defined Units, see above enum.

    Values:
      BYTES: Counter returns a value in bytes.
      BYTES_PER_SEC: Counter returns a value in bytes per second.
      MILLISECONDS: Counter returns a value in milliseconds.
      MICROSECONDS: Counter returns a value in microseconds.
      NANOSECONDS: Counter returns a value in nanoseconds.
      TIMESTAMP_MSEC: Counter returns a timestamp in milliseconds.
      TIMESTAMP_USEC: Counter returns a timestamp in microseconds.
      TIMESTAMP_NSEC: Counter returns a timestamp in nanoseconds.
    """
    BYTES = 0
    BYTES_PER_SEC = 1
    MILLISECONDS = 2
    MICROSECONDS = 3
    NANOSECONDS = 4
    TIMESTAMP_MSEC = 5
    TIMESTAMP_USEC = 6
    TIMESTAMP_NSEC = 7