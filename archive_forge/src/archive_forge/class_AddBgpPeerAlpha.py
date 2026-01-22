from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.networking.routers import routers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.edge_cloud.networking import resource_args
from googlecloudsdk.command_lib.edge_cloud.networking.routers import flags as routers_flags
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AddBgpPeerAlpha(base.UpdateCommand):
    """Add a BGP peer to a Distributed Cloud Edge Network router.

  *{command}* is used to add a BGP peer to a Distributed Cloud Edge Network
  router.
  """
    detailed_help = {'DESCRIPTION': DESCRIPTION, 'EXAMPLES': EXAMPLES_ALPHA}

    @classmethod
    def Args(cls, parser):
        resource_args.AddRouterResourceArg(parser, 'to which we add a bgp peer', True)
        routers_flags.AddBgpPeerArgs(parser, for_update=False, enable_peer_ipv6_range=True)
        base.ASYNC_FLAG.AddToParser(parser)

    def Run(self, args):
        routers_client = routers.RoutersClient(self.ReleaseTrack())
        router_ref = args.CONCEPTS.router.Parse()
        update_req_op = routers_client.AddBgpPeer(router_ref, args)
        async_ = args.async_
        if not async_:
            response = routers_client.WaitForOperation(update_req_op)
            log.UpdatedResource(router_ref.RelativeName(), details='Operation was successful.')
            return response
        log.status.Print('Updating [{0}] with operation [{1}].'.format(router_ref.RelativeName(), update_req_op.name))