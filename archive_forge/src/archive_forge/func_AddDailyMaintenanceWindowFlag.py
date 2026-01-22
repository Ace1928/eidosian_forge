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
def AddDailyMaintenanceWindowFlag(parser, hidden=False, add_unset_text=False):
    """Adds a --maintenance-window flag to parser."""
    help_text = 'Set a time of day when you prefer maintenance to start on this cluster. For example:\n\n  $ {command} example-cluster --maintenance-window=12:43\n\nThe time corresponds to the UTC time zone, and must be in HH:MM format.\n\nNon-emergency maintenance will occur in the 4 hour block starting at the\nspecified time.\n\nThis is mutually exclusive with the recurring maintenance windows\nand will overwrite any existing window. Compatible with maintenance\nexclusions.\n'
    unset_text = "\nTo remove an existing maintenance window from the cluster, use\n'--clear-maintenance-window'.\n"
    description = 'Maintenance windows must be passed in using HH:MM format.'
    unset_description = ' They can also be removed by using the word "None".'
    if add_unset_text:
        help_text += unset_text
        description += unset_description
    type_ = arg_parsers.RegexpValidator('^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$|^None$', description)
    parser.add_argument('--maintenance-window', default=None, hidden=hidden, type=type_, metavar='START_TIME', help=help_text)