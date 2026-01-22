from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def ArgsForSnapshotRef(parser):
    """Register flags for specifying a single Snapshot ID.

  Args:
    parser: The argparse.ArgParser to configure with snapshot arguments.
  """
    parser.add_argument('snapshot', metavar='SNAPSHOT_ID', help='ID of the Cloud Dataflow snapshot.')
    parser.add_argument('--region', required=True, metavar='REGION_ID', help='Region ID of the snapshot regional endpoint.')