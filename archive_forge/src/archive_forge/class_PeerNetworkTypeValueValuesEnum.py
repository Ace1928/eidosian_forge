from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PeerNetworkTypeValueValuesEnum(_messages.Enum):
    """Required. The type of the network to peer with the VMware Engine
    network.

    Values:
      PEER_NETWORK_TYPE_UNSPECIFIED: Unspecified
      STANDARD: Peering connection used for connecting to another VPC network
        established by the same user. For example, a peering connection to
        another VPC network in the same project or to an on-premises network.
      VMWARE_ENGINE_NETWORK: Peering connection used for connecting to another
        VMware Engine network.
      PRIVATE_SERVICES_ACCESS: Peering connection used for establishing
        [private services access](https://cloud.google.com/vpc/docs/private-
        services-access).
      NETAPP_CLOUD_VOLUMES: Peering connection used for connecting to NetApp
        Cloud Volumes.
      THIRD_PARTY_SERVICE: Peering connection used for connecting to third-
        party services. Most third-party services require manual setup of
        reverse peering on the VPC network associated with the third-party
        service.
      DELL_POWERSCALE: Peering connection used for connecting to Dell
        PowerScale Filers
      GOOGLE_CLOUD_NETAPP_VOLUMES: Peering connection used for connecting to
        Google Cloud NetApp Volumes.
    """
    PEER_NETWORK_TYPE_UNSPECIFIED = 0
    STANDARD = 1
    VMWARE_ENGINE_NETWORK = 2
    PRIVATE_SERVICES_ACCESS = 3
    NETAPP_CLOUD_VOLUMES = 4
    THIRD_PARTY_SERVICE = 5
    DELL_POWERSCALE = 6
    GOOGLE_CLOUD_NETAPP_VOLUMES = 7