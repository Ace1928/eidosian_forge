from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_api
from googlecloudsdk.api_lib.network_connectivity import networkconnectivity_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.network_connectivity import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.GA)
class AcceptSpoke(base.Command):
    """Accept a spoke into a hub.

  Accept a proposed or previously rejected VPC spoke. By accepting a spoke,
  you permit connectivity between the associated VPC network
  and other VPC networks that are attached to the same hub.
  """

    @staticmethod
    def Args(parser):
        flags.AddHubResourceArg(parser, 'to accept the spoke into')
        flags.AddSpokeFlag(parser, 'URI of the spoke to accept')
        flags.AddAsyncFlag(parser)

    def Run(self, args):
        client = networkconnectivity_api.HubsClient(release_track=self.ReleaseTrack())
        hub_ref = args.CONCEPTS.hub.Parse()
        op_ref = client.AcceptSpoke(hub_ref, args.spoke)
        log.status.Print('Accept spoke request issued for: [{}]'.format(hub_ref.Name()))
        op_resource = resources.REGISTRY.ParseRelativeName(op_ref.name, collection='networkconnectivity.projects.locations.operations', api_version=networkconnectivity_util.VERSION_MAP[self.ReleaseTrack()])
        poller = waiter.CloudOperationPollerNoResources(client.operation_service)
        if op_ref.done:
            return poller.GetResult(op_resource)
        if args.async_:
            log.status.Print('Check operation [{}] for status.'.format(op_ref.name))
            return op_ref
        res = waiter.WaitFor(poller, op_resource, 'Waiting for operation [{}] to complete'.format(op_ref.name))
        return res