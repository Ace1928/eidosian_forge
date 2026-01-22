from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TypeValueValuesEnum(_messages.Enum):
    """The type of this property.

    Values:
      UNSPECIFIED: The type is unspecified, and will result in an error.
      INT64: The type is `int64`.
      BOOL: The type is `bool`.
      STRING: The type is `string`.
      DOUBLE: The type is 'double'.
    """
    UNSPECIFIED = 0
    INT64 = 1
    BOOL = 2
    STRING = 3
    DOUBLE = 4