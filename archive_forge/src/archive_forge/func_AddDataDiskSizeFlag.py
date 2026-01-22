from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddDataDiskSizeFlag(parser):
    """Adds a --data-disk-size flag to the given parser."""
    help_text = '    Storage capacity available to the database, in GB. The minimum (and\n    default) size is 10GB.\n  '
    parser.add_argument('--data-disk-size', type=int, help=help_text)