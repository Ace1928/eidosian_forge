import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def ToNanoseconds(self):
    """Converts a Duration to nanoseconds."""
    return self.seconds * _NANOS_PER_SECOND + self.nanos