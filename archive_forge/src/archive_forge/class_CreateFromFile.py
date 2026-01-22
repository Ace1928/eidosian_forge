from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import clusters
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class CreateFromFile(base.CreateCommand):
    """Create a cluster from a file."""
    detailed_help = {'EXAMPLES': '\nTo create a cluster from a YAML file, run:\n\n  $ {command} --file=cluster.yaml\n'}

    @classmethod
    def Args(cls, parser):
        parser.add_argument('--file', help='\n        The path to a YAML file containing a Dataproc Cluster resource.\n\n        For more information, see:\n        https://cloud.google.com/dataproc/docs/reference/rest/v1/projects.regions.clusters#Cluster.\n        ', required=True)
        flags.AddTimeoutFlag(parser, default='35m')
        flags.AddRegionFlag(parser)
        base.ASYNC_FLAG.AddToParser(parser)

    def Run(self, args):
        dataproc = dp.Dataproc(self.ReleaseTrack())
        data = console_io.ReadFromFileOrStdin(args.file or '-', binary=False)
        cluster = export_util.Import(message_type=dataproc.messages.Cluster, stream=data)
        cluster_ref = util.ParseCluster(cluster.clusterName, dataproc)
        return clusters.CreateCluster(dataproc, cluster_ref, cluster, args.async_, args.timeout)