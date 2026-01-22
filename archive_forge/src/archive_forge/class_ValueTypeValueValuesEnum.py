from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValueTypeValueValuesEnum(_messages.Enum):
    """Whether the measurement is an integer, a floating-point number, etc.

    Values:
      VALUE_TYPE_UNSPECIFIED: Do not use this default value.
      BOOL: The value is a boolean. This value type can be used only if the
        metric kind is `GAUGE`.
      INT64: The value is a signed 64-bit integer.
      DOUBLE: The value is a double precision floating point number.
      STRING: The value is a text string. This value type can be used only if
        the metric kind is `GAUGE`.
      DISTRIBUTION: The value is a `Distribution`.
      MONEY: The value is money.
    """
    VALUE_TYPE_UNSPECIFIED = 0
    BOOL = 1
    INT64 = 2
    DOUBLE = 3
    STRING = 4
    DISTRIBUTION = 5
    MONEY = 6