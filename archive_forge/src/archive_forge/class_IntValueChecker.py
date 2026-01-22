import ctypes
import numbers
from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf.internal import decoder
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import descriptor
class IntValueChecker(object):
    """Checker used for integer fields.  Performs type-check and range check."""

    def CheckValue(self, proposed_value):
        if not isinstance(proposed_value, numbers.Integral):
            message = '%.1024r has type %s, but expected one of: %s' % (proposed_value, type(proposed_value), (int,))
            raise TypeError(message)
        if not self._MIN <= int(proposed_value) <= self._MAX:
            raise ValueError('Value out of range: %d' % proposed_value)
        proposed_value = int(proposed_value)
        return proposed_value

    def DefaultValue(self):
        return 0