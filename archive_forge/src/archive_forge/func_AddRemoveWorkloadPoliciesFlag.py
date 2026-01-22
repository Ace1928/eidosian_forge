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
def AddRemoveWorkloadPoliciesFlag(parser, hidden=False):
    """Adds Remove workload policies related flags to parser."""
    type_validator = arg_parsers.RegexpValidator('^allow-net-admin$', 'Workload policy only supports "allow-net-admin"')
    help_text = "Remove Autopilot workload policies from the cluster.\n\nExamples:\n\n  $ {command} example-cluster --remove-workload-policies=allow-net-admin\n\nThe only supported workload policy is 'allow-net-admin'.\n"
    parser.add_argument('--remove-workload-policies', type=type_validator, help=help_text, hidden=hidden)