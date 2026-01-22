from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
import six
def GetVpnGatewayForInsert(self, name, description, network, vpn_interfaces_with_interconnect_attachments, stack_type=None, gateway_ip_version=None):
    """Returns the VpnGateway message for an insert request.

    Args:
      name: String representing the name of the VPN Gateway resource.
      description: String representing the description for the VPN Gateway
        resource.
      network: String representing the network URL the VPN gateway resource
        belongs to.
      vpn_interfaces_with_interconnect_attachments: Dict representing pairs
        interface id and interconnected attachment associated with vpn gateway
        on this interface.
      stack_type: Enum presenting the stack type of the vpn gateway resource.
      gateway_ip_version: Enum presenting the gateway IP version of the vpn
        gateway resource.

    Returns:
      The VpnGateway message object that can be used in an insert request.
    """
    target_stack_type = None
    target_gateway_ip_version = None
    if stack_type is not None:
        target_stack_type = self._messages.VpnGateway.StackTypeValueValuesEnum(stack_type)
    if gateway_ip_version is not None:
        target_gateway_ip_version = self._messages.VpnGateway.GatewayIpVersionValueValuesEnum(gateway_ip_version)
    if vpn_interfaces_with_interconnect_attachments is not None:
        vpn_interfaces = []
        for key, value in sorted(vpn_interfaces_with_interconnect_attachments.items()):
            vpn_interfaces.append(self._messages.VpnGatewayVpnGatewayInterface(id=int(key), interconnectAttachment=six.text_type(value)))
        if gateway_ip_version is not None:
            return self._messages.VpnGateway(name=name, description=description, network=network, vpnInterfaces=vpn_interfaces, stackType=target_stack_type, gatewayIpVersion=target_gateway_ip_version)
        return self._messages.VpnGateway(name=name, description=description, network=network, vpnInterfaces=vpn_interfaces, stackType=target_stack_type)
    else:
        if gateway_ip_version is not None:
            return self._messages.VpnGateway(name=name, description=description, network=network, stackType=target_stack_type, gatewayIpVersion=target_gateway_ip_version)
        return self._messages.VpnGateway(name=name, description=description, network=network, stackType=target_stack_type)