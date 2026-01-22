from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def ExtractSnapshotTtlDuration(args):
    """Extract the Duration string for the Snapshot ttl.

  Args:
    args: The command line arguments.
  Returns:
    A duration string for the snapshot ttl.
  """
    return six.text_type(args.snapshot_ttl) + 's'