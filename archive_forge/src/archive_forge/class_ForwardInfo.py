from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ForwardInfo(_messages.Message):
    """Details of the final state "forward" and associated resource.

  Enums:
    TargetValueValuesEnum: Target type where this packet is forwarded to.

  Fields:
    ipAddress: IP address of the target (if applicable).
    resourceUri: URI of the resource that the packet is forwarded to.
    target: Target type where this packet is forwarded to.
  """

    class TargetValueValuesEnum(_messages.Enum):
        """Target type where this packet is forwarded to.

    Values:
      TARGET_UNSPECIFIED: Target not specified.
      PEERING_VPC: Forwarded to a VPC peering network.
      VPN_GATEWAY: Forwarded to a Cloud VPN gateway.
      INTERCONNECT: Forwarded to a Cloud Interconnect connection.
      GKE_MASTER: Forwarded to a Google Kubernetes Engine Container cluster
        master.
      IMPORTED_CUSTOM_ROUTE_NEXT_HOP: Forwarded to the next hop of a custom
        route imported from a peering VPC.
      CLOUD_SQL_INSTANCE: Forwarded to a Cloud SQL instance.
      ANOTHER_PROJECT: Forwarded to a VPC network in another project.
      NCC_HUB: Forwarded to an NCC Hub.
      ROUTER_APPLIANCE: Forwarded to a router appliance.
    """
        TARGET_UNSPECIFIED = 0
        PEERING_VPC = 1
        VPN_GATEWAY = 2
        INTERCONNECT = 3
        GKE_MASTER = 4
        IMPORTED_CUSTOM_ROUTE_NEXT_HOP = 5
        CLOUD_SQL_INSTANCE = 6
        ANOTHER_PROJECT = 7
        NCC_HUB = 8
        ROUTER_APPLIANCE = 9
    ipAddress = _messages.StringField(1)
    resourceUri = _messages.StringField(2)
    target = _messages.EnumField('TargetValueValuesEnum', 3)