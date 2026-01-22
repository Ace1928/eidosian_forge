from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddClusterLabelsFlag(parser):
    """Adds a --cluster-labels flag to the given parser."""
    help_text = '    The resource labels for an AlloyDB cluster. An object containing a list\n    of "key": "value" pairs.\n    '
    parser.add_argument('--cluster-labels', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help=help_text)