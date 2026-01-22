from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetVpnGatewayArgumentForOtherResource(required=False):
    """Returns the flag for specifying the VPN gateway."""
    return compute_flags.ResourceArgument(name='--vpn-gateway', resource_name='VPN Gateway', completer=VpnGatewaysCompleter, plural=False, required=required, regional_collection='compute.vpnGateways', short_help='Reference to a VPN gateway, this flag is used for creating HA VPN tunnels.', region_explanation='Should be the same as region, if not specified, it will be automatically set.', detailed_help='        Reference to a Highly Available VPN gateway.\n        ')