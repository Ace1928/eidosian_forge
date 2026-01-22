from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddPrimaryIdFlag(parser):
    """Adds a --primary-id flag to the given parser."""
    help_text = '    The ID of the primary instance for this AlloyDB cluster.\n    '
    parser.add_argument('--primary-id', help=help_text, required=True)