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
def AddAutoscalingProfilesFlag(parser, hidden=False):
    """Adds autoscaling profiles flag to parser.

  Autoscaling profiles flag is --autoscaling-profile.

  Args:
    parser: A given parser.
    hidden: If true, suppress help text for added options.
  """
    parser.add_argument('--autoscaling-profile', required=False, default=None, help="         Set autoscaling behaviour, choices are 'optimize-utilization' and 'balanced'.\n         Default is 'balanced'.\n      ", hidden=hidden, type=str)