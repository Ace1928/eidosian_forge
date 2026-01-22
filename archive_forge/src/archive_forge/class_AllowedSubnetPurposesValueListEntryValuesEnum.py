from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowedSubnetPurposesValueListEntryValuesEnum(_messages.Enum):
    """AllowedSubnetPurposesValueListEntryValuesEnum enum type.

    Values:
      SUBNET_PURPOSE_CUSTOM_HARDWARE: <no description>
      SUBNET_PURPOSE_PRIVATE: <no description>
      SUBNET_PURPOSE_UNSPECIFIED: <no description>
    """
    SUBNET_PURPOSE_CUSTOM_HARDWARE = 0
    SUBNET_PURPOSE_PRIVATE = 1
    SUBNET_PURPOSE_UNSPECIFIED = 2