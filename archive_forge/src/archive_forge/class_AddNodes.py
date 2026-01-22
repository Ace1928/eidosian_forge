from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.vmware.sddc.clusters import ClustersClient
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.vmware.sddc import flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AddNodes(base.UpdateCommand):
    """Add a node to the cluster in a VMware Engine private cloud."""

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        flags.AddClusterArgToParser(parser)

    def Run(self, args):
        cluster = args.CONCEPTS.cluster.Parse()
        client = ClustersClient()
        operation = client.AddNodes(cluster, 1)
        return client.WaitForOperation(operation, 'waiting for node to be added in the cluster [{}]'.format(cluster))