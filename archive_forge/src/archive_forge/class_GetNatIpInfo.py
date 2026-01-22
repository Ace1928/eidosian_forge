from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.routers import flags as routers_flags
class GetNatIpInfo(base.DescribeCommand):
    """Display NAT IP information in a router."""
    ROUTER_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.ROUTER_ARG = routers_flags.RouterArgument()
        cls.ROUTER_ARG.AddArgument(parser, operation_type='get NAT IP info')
        routers_flags.AddGetNatIpInfoArgs(parser)
        base.URI_FLAG.RemoveFromParser(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        router_ref = self.ROUTER_ARG.ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(client))
        params = router_ref.AsDict()
        if args.nat_name:
            params['natName'] = args.nat_name
        request = client.messages.ComputeRoutersGetNatIpInfoRequest(**params)
        return client.MakeRequests([(client.apitools_client.routers, 'GetNatIpInfo', request)])[0]