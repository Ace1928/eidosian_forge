from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RangeReservation(_messages.Message):
    """Represents a range reservation.

  Fields:
    ipPrefixLength: Required. The size of the desired subnet. Use usual CIDR
      range notation. For example, '29' to find unused x.x.x.x/29 CIDR range.
      The goal is to determine if one of the allocated ranges has enough free
      space for a subnet of the requested size. GCE disallows subnets with
      prefix_length > 29
    requestedRanges: Optional. The name of one or more allocated IP address
      ranges associated with this private service access connection. If no
      range names are provided all ranges associated with this connection will
      be considered. If a CIDR range with the specified IP prefix length is
      not available within these ranges the validation fails.
    secondaryRangeIpPrefixLengths: Optional. The size of the desired secondary
      ranges for the subnet. Use usual CIDR range notation. For example, '29'
      to find unused x.x.x.x/29 CIDR range. The goal is to determine that the
      allocated ranges have enough free space for all the requested secondary
      ranges. GCE disallows subnets with prefix_length > 29
    subnetworkCandidates: Optional. List of subnetwork candidates to validate.
      The required input fields are `name`, `network`, and `region`.
      Subnetworks from this list which exist will be returned in the response
      with the `ip_cidr_range`, `secondary_ip_cider_ranges`, and
      `outside_allocation` fields set.
  """
    ipPrefixLength = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    requestedRanges = _messages.StringField(2, repeated=True)
    secondaryRangeIpPrefixLengths = _messages.IntegerField(3, repeated=True, variant=_messages.Variant.INT32)
    subnetworkCandidates = _messages.MessageField('Subnetwork', 4, repeated=True)