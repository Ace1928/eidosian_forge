from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.snapshots import flags as snap_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _GAArgs(parser):
    """Set Args based on Release Track."""
    parser.add_argument('name', help='The name of the snapshot.')
    snap_flags.AddChainArg(parser)
    snap_flags.AddSourceDiskCsekKey(parser)
    flags.AddGuestFlushFlag(parser, 'snapshot', custom_help='\n  Create an application-consistent snapshot by informing the OS\n  to prepare for the snapshot process. Currently only supported\n  for creating snapshots of disks attached to Windows instances.\n  ')
    flags.AddStorageLocationFlag(parser, 'snapshot')
    labels_util.AddCreateLabelsFlags(parser)
    csek_utils.AddCsekKeyArgs(parser, flags_about_creation=False)
    base.ASYNC_FLAG.AddToParser(parser)
    parser.add_argument('--description', help='Text to describe the new snapshot.')
    snap_flags.SOURCE_DISK_ARG.AddArgument(parser)
    snap_flags.AddSnapshotType(parser)
    snap_flags.SOURCE_DISK_FOR_RECOVERY_CHECKPOINT_ARG.AddArgument(parser)
    snap_flags.SOURCE_INSTANT_SNAPSHOT_ARG.AddArgument(parser)
    snap_flags.AddSourceInstantSnapshotCsekKey(parser)