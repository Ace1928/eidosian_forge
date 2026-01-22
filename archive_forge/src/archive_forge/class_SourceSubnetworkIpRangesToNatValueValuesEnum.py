from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceSubnetworkIpRangesToNatValueValuesEnum(_messages.Enum):
    """Specify the Nat option, which can take one of the following values: -
    ALL_SUBNETWORKS_ALL_IP_RANGES: All of the IP ranges in every Subnetwork
    are allowed to Nat. - ALL_SUBNETWORKS_ALL_PRIMARY_IP_RANGES: All of the
    primary IP ranges in every Subnetwork are allowed to Nat. -
    LIST_OF_SUBNETWORKS: A list of Subnetworks are allowed to Nat (specified
    in the field subnetwork below) The default is
    SUBNETWORK_IP_RANGE_TO_NAT_OPTION_UNSPECIFIED. Note that if this field
    contains ALL_SUBNETWORKS_ALL_IP_RANGES then there should not be any other
    Router.Nat section in any Router for this network in this region.

    Values:
      ALL_SUBNETWORKS_ALL_IP_RANGES: All the IP ranges in every Subnetwork are
        allowed to Nat.
      ALL_SUBNETWORKS_ALL_PRIMARY_IP_RANGES: All the primary IP ranges in
        every Subnetwork are allowed to Nat.
      LIST_OF_SUBNETWORKS: A list of Subnetworks are allowed to Nat (specified
        in the field subnetwork below)
    """
    ALL_SUBNETWORKS_ALL_IP_RANGES = 0
    ALL_SUBNETWORKS_ALL_PRIMARY_IP_RANGES = 1
    LIST_OF_SUBNETWORKS = 2