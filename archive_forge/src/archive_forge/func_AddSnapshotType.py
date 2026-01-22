from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddSnapshotType(parser):
    snapshot_type_choices = sorted(['STANDARD', 'ARCHIVE'])
    parser.add_argument('--snapshot-type', choices=snapshot_type_choices, help='\n              Type of snapshot. If a snapshot type is not specified, a STANDARD snapshot will be created.\n           ')