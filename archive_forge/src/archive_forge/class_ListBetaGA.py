from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute.vpn_tunnels import vpn_tunnels_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.vpn_tunnels import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_projection_spec
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class ListBetaGA(base.ListCommand):
    """List VPN tunnels."""
    detailed_help = None

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat(flags.DEFAULT_LIST_FORMAT)
        lister.AddRegionsArg(parser)
        parser.display_info.AddCacheUpdater(flags.VpnTunnelsCompleter)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        helper = vpn_tunnels_utils.VpnTunnelHelper(holder)
        project = properties.VALUES.core.project.GetOrFail()
        display_info = args.GetDisplayInfo()
        defaults = resource_projection_spec.ProjectionSpec(symbols=display_info.transforms, aliases=display_info.aliases)
        args.filter, filter_expr = filter_rewrite.Rewriter().Rewrite(args.filter, defaults=defaults)
        return helper.List(project=project, filter_expr=filter_expr)