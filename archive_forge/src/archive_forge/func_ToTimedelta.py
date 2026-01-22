import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def ToTimedelta(self):
    """Converts Duration to timedelta."""
    return datetime.timedelta(seconds=self.seconds, microseconds=_RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND))