from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SupportedByValueListEntryValuesEnum(_messages.Enum):
    """SupportedByValueListEntryValuesEnum enum type.

    Values:
      ENUM_TYPE_UNSPECIFIED: Unused.
      INSPECT: Supported by the inspect operations.
      RISK_ANALYSIS: Supported by the risk analysis operations.
    """
    ENUM_TYPE_UNSPECIFIED = 0
    INSPECT = 1
    RISK_ANALYSIS = 2