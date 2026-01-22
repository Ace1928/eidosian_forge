from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExternalVpnGatewayInterface(_messages.Message):
    """The interface for the external VPN gateway.

  Fields:
    id: The numeric ID of this interface. The allowed input values for this id
      for different redundancy types of external VPN gateway: -
      SINGLE_IP_INTERNALLY_REDUNDANT - 0 - TWO_IPS_REDUNDANCY - 0, 1 -
      FOUR_IPS_REDUNDANCY - 0, 1, 2, 3
    ipAddress: IP address of the interface in the external VPN gateway. Only
      IPv4 is supported. This IP address can be either from your on-premise
      gateway or another Cloud provider's VPN gateway, it cannot be an IP
      address from Google Compute Engine.
    ipv6Address: IPv6 address of the interface in the external VPN gateway.
      This IPv6 address can be either from your on-premise gateway or another
      Cloud provider's VPN gateway, it cannot be an IP address from Google
      Compute Engine. Must specify an IPv6 address (not IPV4-mapped) using any
      format described in RFC 4291 (e.g. 2001:db8:0:0:2d9:51:0:0). The output
      format is RFC 5952 format (e.g. 2001:db8::2d9:51:0:0).
  """
    id = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    ipAddress = _messages.StringField(2)
    ipv6Address = _messages.StringField(3)