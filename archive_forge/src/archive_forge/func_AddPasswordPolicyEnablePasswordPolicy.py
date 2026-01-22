from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddPasswordPolicyEnablePasswordPolicy(parser, show_negated_in_help=False, hidden=False):
    """Add the flag to enable password policy.

  Args:
    parser: The current argparse parser to add this to.
    show_negated_in_help: Show nagative action in help.
    hidden: if the field needs to be hidden.
  """
    kwargs = _GetKwargsForBoolFlag(show_negated_in_help)
    parser.add_argument('--enable-password-policy', required=False, help='        Enable the password policy, which enforces user password management with\n        the policies configured for the instance. This flag is only available for Postgres.\n      ', hidden=hidden, **kwargs)