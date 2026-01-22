from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddTreeTypeFlag(parser, required=True, default_value='DRAFT'):
    """Adds the --tree-type flag to the given parser."""
    help_text = '    Tree type for database entities.\n    '
    choices = ['SOURCE', 'DRAFT']
    parser.add_argument('--tree-type', help=help_text, choices=choices, required=required, default=default_value)