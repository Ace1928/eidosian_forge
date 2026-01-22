from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1VpcSubNetwork(_messages.Message):
    """Sub-segment ranges inside of a VPC Network.

  Fields:
    network: Required. Network name. If the network is not part of the
      organization, the `compute.network.get` permission must be granted to
      the caller. Format: `//compute.googleapis.com/projects/{PROJECT_ID}/glob
      al/networks/{NETWORK_NAME}` Example:
      `//compute.googleapis.com/projects/my-project/global/networks/network-1`
    vpcIpSubnetworks: CIDR block IP subnetwork specification. The IP address
      must be an IPv4 address and can be a public or private IP address. Note
      that for a CIDR IP address block, the specified IP address portion must
      be properly truncated (i.e. all the host bits must be zero) or the input
      is considered malformed. For example, "192.0.2.0/24" is accepted but
      "192.0.2.1/24" is not. If empty, all IP addresses are allowed.
  """
    network = _messages.StringField(1)
    vpcIpSubnetworks = _messages.StringField(2, repeated=True)