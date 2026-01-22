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
def AddStartIpRotationFlag(parser, hidden=False):
    """Adds a --start-ip-rotation flag to parser."""
    help_text = 'Start the rotation of this cluster to a new IP. For example:\n\n  $ {command} example-cluster --start-ip-rotation\n\nThis causes the cluster to serve on two IPs, and will initiate a node upgrade to point to the new IP. See documentation for more details: https://cloud.google.com/kubernetes-engine/docs/how-to/ip-rotation.'
    parser.add_argument('--start-ip-rotation', action='store_true', default=False, hidden=hidden, help=help_text)