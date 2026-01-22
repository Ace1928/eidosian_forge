from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NatInfo(_messages.Message):
    """For display only. Metadata associated with NAT.

  Enums:
    TypeValueValuesEnum: Type of NAT.

  Fields:
    natGatewayName: The name of Cloud NAT Gateway. Only valid when type is
      CLOUD_NAT.
    networkUri: URI of the network where NAT translation takes place.
    newDestinationIp: Destination IP address after NAT translation.
    newDestinationPort: Destination port after NAT translation. Only valid
      when protocol is TCP or UDP.
    newSourceIp: Source IP address after NAT translation.
    newSourcePort: Source port after NAT translation. Only valid when protocol
      is TCP or UDP.
    oldDestinationIp: Destination IP address before NAT translation.
    oldDestinationPort: Destination port before NAT translation. Only valid
      when protocol is TCP or UDP.
    oldSourceIp: Source IP address before NAT translation.
    oldSourcePort: Source port before NAT translation. Only valid when
      protocol is TCP or UDP.
    protocol: IP protocol in string format, for example: "TCP", "UDP", "ICMP".
    routerUri: Uri of the Cloud Router. Only valid when type is CLOUD_NAT.
    type: Type of NAT.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of NAT.

    Values:
      TYPE_UNSPECIFIED: Type is unspecified.
      INTERNAL_TO_EXTERNAL: From Compute Engine instance's internal address to
        external address.
      EXTERNAL_TO_INTERNAL: From Compute Engine instance's external address to
        internal address.
      CLOUD_NAT: Cloud NAT Gateway.
      PRIVATE_SERVICE_CONNECT: Private service connect NAT.
    """
        TYPE_UNSPECIFIED = 0
        INTERNAL_TO_EXTERNAL = 1
        EXTERNAL_TO_INTERNAL = 2
        CLOUD_NAT = 3
        PRIVATE_SERVICE_CONNECT = 4
    natGatewayName = _messages.StringField(1)
    networkUri = _messages.StringField(2)
    newDestinationIp = _messages.StringField(3)
    newDestinationPort = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    newSourceIp = _messages.StringField(5)
    newSourcePort = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    oldDestinationIp = _messages.StringField(7)
    oldDestinationPort = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    oldSourceIp = _messages.StringField(9)
    oldSourcePort = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    protocol = _messages.StringField(11)
    routerUri = _messages.StringField(12)
    type = _messages.EnumField('TypeValueValuesEnum', 13)