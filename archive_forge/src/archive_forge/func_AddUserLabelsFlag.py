from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddUserLabelsFlag(parser):
    """Adds a --user-labels flag to the given parser."""
    help_text = '    The resource labels for a Cloud SQL instance to use to annotate any related\n    underlying resources such as Compute Engine VMs. An object containing a list\n    of "key": "value" pairs.\n    '
    parser.add_argument('--user-labels', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help=help_text)