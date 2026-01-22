import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def Unpack(self, msg):
    """Unpacks the current Any message into specified message."""
    descriptor = msg.DESCRIPTOR
    if not self.Is(descriptor):
        return False
    msg.ParseFromString(self.value)
    return True