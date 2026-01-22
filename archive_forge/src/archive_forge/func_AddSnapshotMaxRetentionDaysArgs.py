from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def AddSnapshotMaxRetentionDaysArgs(parser, required=True):
    """Adds max retention days flag for snapshot schedule resource policies."""
    parser.add_argument('--max-retention-days', required=required, type=arg_parsers.BoundedInt(lower_bound=1), help='Maximum number of days snapshot can be retained.')