from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.core.resource import resource_transform
import six
def TransformVpnTunnelGateway(vpn_tunnel, undefined=''):
    """Returns the gateway for the specified VPN tunnel resource if applicable.

  Args:
    vpn_tunnel: JSON-serializable object of a VPN tunnel.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    The VPN gateway information in the VPN tunnel object.
  """
    target_vpn_gateway = resource_transform.GetKeyValue(vpn_tunnel, 'targetVpnGateway', None)
    if target_vpn_gateway is not None:
        return target_vpn_gateway
    vpn_gateway = resource_transform.GetKeyValue(vpn_tunnel, 'vpnGateway', None)
    if vpn_gateway is not None:
        return vpn_gateway
    return undefined