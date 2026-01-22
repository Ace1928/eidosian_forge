from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudServicenetworkingV1ConsumerConfigReservedRange(_messages.Message):
    """Allocated IP address ranges for this private service access connection.

  Fields:
    address: The starting address of the reserved range. The address must be a
      valid IPv4 address in the x.x.x.x format. This value combined with the
      IP prefix length is the CIDR range for the reserved range.
    ipPrefixLength: The prefix length of the reserved range.
    name: The name of the reserved range.
  """
    address = _messages.StringField(1)
    ipPrefixLength = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    name = _messages.StringField(3)