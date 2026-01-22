import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def FromSeconds(self, seconds):
    """Converts seconds to Duration."""
    self.seconds = seconds
    self.nanos = 0