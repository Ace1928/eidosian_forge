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
def AddCompleteCredentialRotationFlag(parser, hidden=False):
    """Adds a --complete-credential-rotation flag to parser."""
    help_text = 'Complete the IP and credential rotation for this cluster. For example:\n\n  $ {command} example-cluster --complete-credential-rotation\n\nThis causes the cluster to stop serving its old IP, return to a single IP, and invalidate old credentials. See documentation for more details: https://cloud.google.com/kubernetes-engine/docs/how-to/credential-rotation.'
    parser.add_argument('--complete-credential-rotation', action='store_true', default=False, hidden=hidden, help=help_text)