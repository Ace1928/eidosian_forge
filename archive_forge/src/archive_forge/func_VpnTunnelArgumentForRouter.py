from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def VpnTunnelArgumentForRouter(required=True, operation_type='added'):
    return compute_flags.ResourceArgument(resource_name='vpn tunnel', name='--vpn-tunnel', completer=VpnTunnelsCompleter, plural=False, required=required, regional_collection='compute.vpnTunnels', short_help='The tunnel of the interface being {0}.'.format(operation_type), region_explanation='If not specified it will be set to the region of the router.')