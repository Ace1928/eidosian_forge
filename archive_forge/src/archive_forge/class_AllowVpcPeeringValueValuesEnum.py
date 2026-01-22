from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowVpcPeeringValueValuesEnum(_messages.Enum):
    """Specifies whether VPC peering is allowed.

    Values:
      VPC_PEERING_ALLOWED: <no description>
      VPC_PEERING_BLOCKED: <no description>
      VPC_PEERING_UNSPECIFIED: <no description>
    """
    VPC_PEERING_ALLOWED = 0
    VPC_PEERING_BLOCKED = 1
    VPC_PEERING_UNSPECIFIED = 2