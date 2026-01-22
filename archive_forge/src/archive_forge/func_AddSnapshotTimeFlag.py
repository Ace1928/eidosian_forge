from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
def AddSnapshotTimeFlag(parser):
    """Adds flag for snapshot time to the given parser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--snapshot-time', metavar='SNAPSHOT_TIME', type=str, default=None, required=False, help="\n      The version of the database to export.\n\n      The timestamp must be in the past, rounded to the minute and not older\n      than `earliestVersionTime`. If specified, then the exported documents will\n      represent a consistent view of the database at the provided time.\n      Otherwise, there are no guarantees about the consistency of the exported\n      documents.\n\n      For example, to operate on snapshot time `2023-05-26T10:20:00.00Z`:\n\n        $ {command} --snapshot-time='2023-05-26T10:20:00.00Z'\n      ")