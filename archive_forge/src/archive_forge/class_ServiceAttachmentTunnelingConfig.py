from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceAttachmentTunnelingConfig(_messages.Message):
    """Use to configure this PSC connection in tunneling mode. In tunneling
  mode traffic from consumer to producer will be encapsulated as it crosses
  the VPC boundary and traffic from producer to consumer will be decapsulated
  in the same manner.

  Enums:
    EncapsulationProfileValueValuesEnum: Specify the encapsulation protocol
      and what metadata to include in incoming encapsulated packet headers.
    RoutingModeValueValuesEnum: How this Service Attachment will treat traffic
      sent to the tunnel_ip, destined for the consumer network.

  Fields:
    encapsulationProfile: Specify the encapsulation protocol and what metadata
      to include in incoming encapsulated packet headers.
    routingMode: How this Service Attachment will treat traffic sent to the
      tunnel_ip, destined for the consumer network.
  """

    class EncapsulationProfileValueValuesEnum(_messages.Enum):
        """Specify the encapsulation protocol and what metadata to include in
    incoming encapsulated packet headers.

    Values:
      GENEVE_SECURITY_V1: Use GENEVE encapsulation protocol and include the
        SECURITY_V1 set of GENEVE headers.
      UNSPECIFIED_ENCAPSULATION_PROFILE: <no description>
    """
        GENEVE_SECURITY_V1 = 0
        UNSPECIFIED_ENCAPSULATION_PROFILE = 1

    class RoutingModeValueValuesEnum(_messages.Enum):
        """How this Service Attachment will treat traffic sent to the tunnel_ip,
    destined for the consumer network.

    Values:
      PACKET_INJECTION: Traffic sent to this service attachment will be
        reinjected into the consumer network.
      STANDARD_ROUTING: Response traffic, after de-encapsulation, will be
        returned to the client.
      UNSPECIFIED_ROUTING_MODE: <no description>
    """
        PACKET_INJECTION = 0
        STANDARD_ROUTING = 1
        UNSPECIFIED_ROUTING_MODE = 2
    encapsulationProfile = _messages.EnumField('EncapsulationProfileValueValuesEnum', 1)
    routingMode = _messages.EnumField('RoutingModeValueValuesEnum', 2)