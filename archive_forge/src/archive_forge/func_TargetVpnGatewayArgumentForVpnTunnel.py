from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def TargetVpnGatewayArgumentForVpnTunnel(required=True):
    return compute_flags.ResourceArgument(resource_name='Target VPN Gateway', name='--target-vpn-gateway', completer=TargetVpnGatewaysCompleter, plural=False, required=required, regional_collection='compute.targetVpnGateways', short_help='A reference to a Cloud VPN Classic Target VPN Gateway.', region_explanation='Should be the same as region, if not specified, it will be automatically set.')