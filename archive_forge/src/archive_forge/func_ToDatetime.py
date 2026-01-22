import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def ToDatetime(self, tzinfo=None):
    """Converts Timestamp to a datetime.

    Args:
      tzinfo: A datetime.tzinfo subclass; defaults to None.

    Returns:
      If tzinfo is None, returns a timezone-naive UTC datetime (with no timezone
      information, i.e. not aware that it's UTC).

      Otherwise, returns a timezone-aware datetime in the input timezone.
    """
    _CheckTimestampValid(self.seconds, self.nanos)
    delta = datetime.timedelta(seconds=self.seconds, microseconds=_RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND))
    if tzinfo is None:
        return _EPOCH_DATETIME_NAIVE + delta
    else:
        return (_EPOCH_DATETIME_AWARE + delta).astimezone(tzinfo)