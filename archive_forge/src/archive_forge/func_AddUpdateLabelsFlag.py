from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddUpdateLabelsFlag(parser):
    """Adds Update Labels related flags to parser.

  Args:
    parser: A given parser.
  """
    help_text = 'Labels to apply to the Google Cloud resources in use by the Kubernetes Engine\ncluster. These are unrelated to Kubernetes labels.\n\nExamples:\n\n  $ {command} example-cluster --update-labels=label_a=value1,label_b=value2\n'
    parser.add_argument('--update-labels', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help=help_text)