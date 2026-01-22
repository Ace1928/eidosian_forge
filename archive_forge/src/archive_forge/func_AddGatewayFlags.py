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
def AddGatewayFlags(parser, hidden=False):
    """Adds --gateway-api flag to the parser.

  Flag:
  --gateway-api={disabled|standard},

  Args:
    parser: A given parser.
    hidden: Indicates that the flags are hidden.
  """
    help_text = '\nEnables GKE Gateway controller in this cluster. The value of the flag specifies\nwhich Open Source Gateway API release channel will be used to define Gateway\nresources.\n'
    parser.add_argument('--gateway-api', help=help_text, required=False, choices={'disabled': '              Gateway controller will be disabled in the cluster.\n              ', 'standard': '              Gateway controller will be enabled in the cluster.\n              Resource definitions from the `standard` OSS Gateway API release\n              channel will be installed. '}, default=None, hidden=hidden)