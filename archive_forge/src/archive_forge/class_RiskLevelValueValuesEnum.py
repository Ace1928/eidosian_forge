from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RiskLevelValueValuesEnum(_messages.Enum):
    """The risk level selected for the scan

    Values:
      RISK_LEVEL_UNSPECIFIED: Use default, which is NORMAL.
      NORMAL: Normal scanning (Recommended)
      LOW: Lower impact scanning
    """
    RISK_LEVEL_UNSPECIFIED = 0
    NORMAL = 1
    LOW = 2