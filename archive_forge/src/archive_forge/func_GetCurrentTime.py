import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def GetCurrentTime(self):
    """Get the current UTC into Timestamp."""
    self.FromDatetime(datetime.datetime.utcnow())