from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsableSubnetwork(_messages.Message):
    """UsableSubnetwork resource returns the subnetwork name, its associated
  network and the primary CIDR range.

  Fields:
    ipCidrRange: The range of internal addresses that are owned by this
      subnetwork.
    network: Network Name. Example: projects/my-project/global/networks/my-
      network
    secondaryIpRanges: Secondary IP ranges.
    statusMessage: A human readable status message representing the reasons
      for cases where the caller cannot use the secondary ranges under the
      subnet. For example if the secondary_ip_ranges is empty due to a
      permission issue, an insufficient permission message will be given by
      status_message.
    subnetwork: Subnetwork Name. Example: projects/my-project/regions/us-
      central1/subnetworks/my-subnet
  """
    ipCidrRange = _messages.StringField(1)
    network = _messages.StringField(2)
    secondaryIpRanges = _messages.MessageField('UsableSubnetworkSecondaryRange', 3, repeated=True)
    statusMessage = _messages.StringField(4)
    subnetwork = _messages.StringField(5)