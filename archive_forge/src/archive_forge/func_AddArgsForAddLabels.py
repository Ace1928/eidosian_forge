from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddArgsForAddLabels(parser, required=True):
    """Adds the required --labels flags for add-labels command."""
    required_labels_flag = base.Argument('--labels', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), action=arg_parsers.UpdateAction, required=required, help='A list of labels to add.')
    required_labels_flag.AddToParser(parser)