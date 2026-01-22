from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceLevelValueValuesEnum(_messages.Enum):
    """Output only. Service level of the volume

    Values:
      SERVICE_LEVEL_UNSPECIFIED: Unspecified service level.
      PREMIUM: Premium service level.
      EXTREME: Extreme service level.
      STANDARD: Standard (Software offering)
    """
    SERVICE_LEVEL_UNSPECIFIED = 0
    PREMIUM = 1
    EXTREME = 2
    STANDARD = 3