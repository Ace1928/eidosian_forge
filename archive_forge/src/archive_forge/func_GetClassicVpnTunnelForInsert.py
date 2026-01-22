from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def GetClassicVpnTunnelForInsert(self, name, description, ike_version, peer_ip, shared_secret, target_vpn_gateway, router=None, local_traffic_selector=None, remote_traffic_selector=None):
    """Returns the Classic VpnTunnel message for an insert request.

    Args:
      name: String representing the name of the VPN tunnel resource.
      description: String representing the description for the VPN tunnel
        resource.
      ike_version: The IKE protocol version for establishing the VPN tunnel.
      peer_ip: String representing the peer IP address for the VPN tunnel.
      shared_secret: String representing the shared secret (IKE pre-shared key).
      target_vpn_gateway: String representing the Target VPN Gateway URL the VPN
        tunnel resource should be associated with.
      router: String representing the Router URL the VPN tunnel resource should
        be associated with.
      local_traffic_selector: List of strings representing the local CIDR ranges
        that should be able to send traffic using this VPN tunnel.
      remote_traffic_selector: List of strings representing the remote CIDR
        ranges that should be able to send traffic using this VPN tunnel.

    Returns:
      The VpnTunnel message object that can be used in an insert request.
    """
    return self._messages.VpnTunnel(name=name, description=description, ikeVersion=ike_version, peerIp=peer_ip, sharedSecret=shared_secret, targetVpnGateway=target_vpn_gateway, router=router, localTrafficSelector=local_traffic_selector or [], remoteTrafficSelector=remote_traffic_selector or [])