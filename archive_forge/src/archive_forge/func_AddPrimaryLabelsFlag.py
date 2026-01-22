from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddPrimaryLabelsFlag(parser):
    """Adds a --primary-labels flag to the given parser."""
    help_text = '    The resource labels for an AlloyDB primary instance. An object containing a\n    list of "key": "value" pairs.\n    '
    parser.add_argument('--primary-labels', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help=help_text)