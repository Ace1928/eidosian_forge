import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def FromNanoseconds(self, nanos):
    """Converts nanoseconds to Duration."""
    self._NormalizeDuration(nanos // _NANOS_PER_SECOND, nanos % _NANOS_PER_SECOND)