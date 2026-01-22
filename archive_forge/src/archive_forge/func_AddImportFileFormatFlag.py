from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddImportFileFormatFlag(parser):
    """Adds the --file-format flag to the given parser."""
    help_text = '    File format type to import rules from.\n    '
    choices = ['ORA2PG']
    parser.add_argument('--file-format', help=help_text, choices=choices, required=True)