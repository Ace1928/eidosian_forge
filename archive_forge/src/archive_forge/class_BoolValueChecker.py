import ctypes
import numbers
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import wire_format
from google.protobuf import descriptor
class BoolValueChecker(object):
    """Type checker used for bool fields."""

    def CheckValue(self, proposed_value):
        if not hasattr(proposed_value, '__index__') or (type(proposed_value).__module__ == 'numpy' and type(proposed_value).__name__ == 'ndarray'):
            message = '%.1024r has type %s, but expected one of: %s' % (proposed_value, type(proposed_value), (bool, int))
            raise TypeError(message)
        return bool(proposed_value)

    def DefaultValue(self):
        return False