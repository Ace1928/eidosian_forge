from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpnGatewayStatusTunnel(_messages.Message):
    """Contains some information about a VPN tunnel.

  Fields:
    localGatewayInterface: The VPN gateway interface this VPN tunnel is
      associated with.
    peerGatewayInterface: The peer gateway interface this VPN tunnel is
      connected to, the peer gateway could either be an external VPN gateway
      or a Google Cloud VPN gateway.
    tunnelUrl: URL reference to the VPN tunnel.
  """
    localGatewayInterface = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    peerGatewayInterface = _messages.IntegerField(2, variant=_messages.Variant.UINT32)
    tunnelUrl = _messages.StringField(3)