import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def _CheckTimestampValid(seconds, nanos):
    if seconds < _TIMESTAMP_SECONDS_MIN or seconds > _TIMESTAMP_SECONDS_MAX:
        raise ValueError('Timestamp is not valid: Seconds {0} must be in range [-62135596800, 253402300799].'.format(seconds))
    if nanos < 0 or nanos >= _NANOS_PER_SECOND:
        raise ValueError('Timestamp is not valid: Nanos {} must be in a range [0, 999999].'.format(nanos))