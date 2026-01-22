from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _CreateIapTunnelHelper(self, args, target):
    if self.support_security_gateway and args.security_gateway:
        tunneler = iap_tunnel.SecurityGatewayTunnelHelper(args, project=target.project, region=target.region, security_gateway=target.security_gateway, host=target.host, port=target.port)
    elif target.host:
        tunneler = iap_tunnel.IAPWebsocketTunnelHelper(args, target.project, region=target.region, network=target.network, host=target.host, port=target.port, dest_group=target.dest_group)
    else:
        tunneler = iap_tunnel.IAPWebsocketTunnelHelper(args, target.project, zone=target.zone, instance=target.instance, interface=target.interface, port=target.port)
    if args.listen_on_stdin:
        iap_tunnel_helper = iap_tunnel.IapTunnelStdinHelper(tunneler)
    else:
        local_host, local_port = self._GetLocalHostPort(args)
        check_connection = True
        if hasattr(args, 'iap_tunnel_disable_connection_check'):
            check_connection = not args.iap_tunnel_disable_connection_check
        iap_tunnel_helper = iap_tunnel.IapTunnelProxyServerHelper(local_host, local_port, check_connection, tunneler)
    return iap_tunnel_helper