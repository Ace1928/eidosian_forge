from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DtypeValueValuesEnum(_messages.Enum):
    """The data type of tensor.

    Values:
      DATA_TYPE_UNSPECIFIED: Not a legal value for DataType. Used to indicate
        a DataType field has not been set.
      BOOL: Data types that all computation devices are expected to be capable
        to support.
      STRING: <no description>
      FLOAT: <no description>
      DOUBLE: <no description>
      INT8: <no description>
      INT16: <no description>
      INT32: <no description>
      INT64: <no description>
      UINT8: <no description>
      UINT16: <no description>
      UINT32: <no description>
      UINT64: <no description>
    """
    DATA_TYPE_UNSPECIFIED = 0
    BOOL = 1
    STRING = 2
    FLOAT = 3
    DOUBLE = 4
    INT8 = 5
    INT16 = 6
    INT32 = 7
    INT64 = 8
    UINT8 = 9
    UINT16 = 10
    UINT32 = 11
    UINT64 = 12