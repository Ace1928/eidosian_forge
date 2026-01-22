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
def AddCompleteConvertToAutopilotFlag(parser, hidden=True):
    """Adds --complete-convert-to-autopilot flag to parser."""
    help_text = 'Commit the Autopilot conversion operation by deleting all Standard node pools\nand completing CA rotation. This action requires that a conversion has been\nstarted and that workload migration has completed, with no pods running on GKE\nStandard node pools.\n\nThis action will be automatically performed 72 hours after conversion.\n'
    parser.add_argument('--complete-convert-to-autopilot', default=None, help=help_text, action='store_true', hidden=hidden)