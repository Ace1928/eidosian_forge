from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecondaryIpRangeSpec(_messages.Message):
    """A SecondaryIpRangeSpec object.

  Fields:
    ipPrefixLength: Required. The prefix length of the secondary IP range. Use
      CIDR range notation, such as `30` to provision a secondary IP range with
      an `x.x.x.x/30` CIDR range. The IP address range is drawn from a pool of
      available ranges in the service consumer's allocated range.
    outsideAllocationPublicIpRange: Optional. Enable outside allocation using
      public IP addresses. Any public IP range may be specified. If this field
      is provided, we will not use customer reserved ranges for this secondary
      IP range.
    rangeName: Required. A name for the secondary IP range. The name must be
      1-63 characters long, and comply with RFC1035. The name must be unique
      within the subnetwork.
    requestedAddress: Optional. The starting address of a range. The address
      must be a valid IPv4 address in the x.x.x.x format. This value combined
      with the IP prefix range is the CIDR range for the secondary IP range.
      The range must be within the allocated range that is assigned to the
      private connection. If the CIDR range isn't available, the call fails.
  """
    ipPrefixLength = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    outsideAllocationPublicIpRange = _messages.StringField(2)
    rangeName = _messages.StringField(3)
    requestedAddress = _messages.StringField(4)