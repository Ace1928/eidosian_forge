from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EgressValueValuesEnum(_messages.Enum):
    """Optional. Traffic VPC egress settings. If not provided, it defaults to
    PRIVATE_RANGES_ONLY.

    Values:
      VPC_EGRESS_UNSPECIFIED: Unspecified
      ALL_TRAFFIC: All outbound traffic is routed through the VPC connector.
      PRIVATE_RANGES_ONLY: Only private IP ranges are routed through the VPC
        connector.
    """
    VPC_EGRESS_UNSPECIFIED = 0
    ALL_TRAFFIC = 1
    PRIVATE_RANGES_ONLY = 2