from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def ExtractSnapshotRef(args):
    """Extract the Snapshot Ref for a command. Used with ArgsForSnapshotRef.

  Args:
    args: The command line arguments.
  Returns:
    A Snapshot resource.
  """
    snapshot = args.snapshot
    region = dataflow_util.GetRegion(args)
    return resources.REGISTRY.Parse(snapshot, params={'projectId': properties.VALUES.core.project.GetOrFail, 'location': region}, collection='dataflow.projects.locations.snapshots')