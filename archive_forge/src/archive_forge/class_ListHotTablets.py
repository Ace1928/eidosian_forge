from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core.util import times
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ListHotTablets(base.ListCommand):
    """List hot tablets in a Cloud Bigtable cluster."""
    detailed_help = {'EXAMPLES': textwrap.dedent('            Search for hot tablets in the past 24 hours:\n\n              $ {command} my-cluster-id --instance=my-instance-id\n\n            Search for hot tablets with start and end times by minute:\n\n              $ {command} my-cluster-id --instance=my-instance-id --start-time="2018-08-12 03:30:00" --end-time="2018-08-13 17:00:00"\n\n            Search for hot tablets with start and end times by day:\n\n              $ {command} my-cluster-id --instance=my-instance-id --start-time=2018-01-01 --end-time=2018-01-05\n          ')}

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        arguments.AddClusterResourceArg(parser, 'to list hot tablets for')
        arguments.AddStartTimeArgs(parser, 'to search for hot tablets')
        arguments.AddEndTimeArgs(parser, 'to search for hot tablets')
        parser.display_info.AddFormat("\n      table(\n        tableName.basename():label=TABLE,\n        nodeCpuUsagePercent:label=CPU_USAGE:sort=1:reverse,\n        startTime.date('%Y-%m-%dT%H:%M:%S%Oz', undefined='-'):label=START_TIME,\n        endTime.date('%Y-%m-%dT%H:%M:%S%Oz', undefined='-'):label=END_TIME,\n        startKey:label=START_KEY,\n        endKey:label=END_KEY\n      )\n    ")

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Yields:
      Some value that we want to have printed later.
    """
        cli = util.GetAdminClient()
        cluster_ref = args.CONCEPTS.cluster.Parse()
        msg = util.GetAdminMessages().BigtableadminProjectsInstancesClustersHotTabletsListRequest(parent=cluster_ref.RelativeName(), startTime=args.start_time and times.FormatDateTime(args.start_time), endTime=args.end_time and times.FormatDateTime(args.end_time))
        for hot_tablet in list_pager.YieldFromList(cli.projects_instances_clusters_hotTablets, msg, field='hotTablets', batch_size_attribute=None):
            yield hot_tablet