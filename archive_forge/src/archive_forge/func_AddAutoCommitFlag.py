from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddAutoCommitFlag(parser):
    """Adds a --auto-commit flag to the given parser."""
    help_text = 'Auto commits the conversion workspace.'
    parser.add_argument('--auto-commit', action='store_true', help=help_text)