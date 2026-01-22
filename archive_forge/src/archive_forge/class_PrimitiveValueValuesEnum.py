from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrimitiveValueValuesEnum(_messages.Enum):
    """Primitive types: `true`, `1u`, `-2.0`, `'string'`, `b'bytes'`.

    Values:
      PRIMITIVE_TYPE_UNSPECIFIED: Unspecified type.
      BOOL: Boolean type.
      INT64: Int64 type. Proto-based integer values are widened to int64.
      UINT64: Uint64 type. Proto-based unsigned integer values are widened to
        uint64.
      DOUBLE: Double type. Proto-based float values are widened to double
        values.
      STRING: String type.
      BYTES: Bytes type.
    """
    PRIMITIVE_TYPE_UNSPECIFIED = 0
    BOOL = 1
    INT64 = 2
    UINT64 = 3
    DOUBLE = 4
    STRING = 5
    BYTES = 6