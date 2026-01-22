from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpokeTypeValueValuesEnum(_messages.Enum):
    """Output only. The type of the spokes.

    Values:
      SPOKE_TYPE_UNSPECIFIED: Unspecified spoke type.
      VPN_TUNNEL: Spokes associated with VPN tunnels.
      INTERCONNECT_ATTACHMENT: Spokes associated with VLAN attachments.
      ROUTER_APPLIANCE: Spokes associated with router appliance instances.
      VPC_NETWORK: Spokes associated with VPC networks.
    """
    SPOKE_TYPE_UNSPECIFIED = 0
    VPN_TUNNEL = 1
    INTERCONNECT_ATTACHMENT = 2
    ROUTER_APPLIANCE = 3
    VPC_NETWORK = 4