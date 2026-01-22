from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowAutoModeSubnetValueValuesEnum(_messages.Enum):
    """Specifies whether auto mode subnet creation is allowed.

    Values:
      AUTO_MODE_SUBNET_ALLOWED: <no description>
      AUTO_MODE_SUBNET_BLOCKED: <no description>
      AUTO_MODE_SUBNET_UNSPECIFIED: <no description>
    """
    AUTO_MODE_SUBNET_ALLOWED = 0
    AUTO_MODE_SUBNET_BLOCKED = 1
    AUTO_MODE_SUBNET_UNSPECIFIED = 2