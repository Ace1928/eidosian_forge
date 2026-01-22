from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def VpnTunnelArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(resource_name='VPN Tunnel', completer=VpnTunnelsCompleter, plural=plural, required=required, regional_collection='compute.vpnTunnels', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)