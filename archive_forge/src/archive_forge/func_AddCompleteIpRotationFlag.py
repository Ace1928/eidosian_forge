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
def AddCompleteIpRotationFlag(parser, hidden=False):
    """Adds a --complete-ip-rotation flag to parser."""
    help_text = 'Complete the IP rotation for this cluster. For example:\n\n  $ {command} example-cluster --complete-ip-rotation\n\nThis causes the cluster to stop serving its old IP, and return to a single IP state. See documentation for more details: https://cloud.google.com/kubernetes-engine/docs/how-to/ip-rotation.'
    parser.add_argument('--complete-ip-rotation', action='store_true', default=False, hidden=hidden, help=help_text)