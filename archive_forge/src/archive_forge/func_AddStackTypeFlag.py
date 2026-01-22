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
def AddStackTypeFlag(parser, hidden=False):
    """Adds --stack-type flag to the given parser.

  Args:
    parser: A given parser.
    hidden: If true, suppress help text for added options.
  """
    help_text = 'IP stack type of the node VMs.'
    parser.add_argument('--stack-type', hidden=hidden, help=help_text, choices=['ipv4', 'ipv4-ipv6'])