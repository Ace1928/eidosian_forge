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
def AddPasswordPolicyDisallowUsernameSubstring(parser, show_negated_in_help=True, hidden=False):
    """Add the flag to specify password policy disallow username as substring.

  Args:
    parser: The current argparse parser to add this to.
    show_negated_in_help: Show nagative action in help.
    hidden: if the field needs to be hidden.
  """
    kwargs = _GetKwargsForBoolFlag(show_negated_in_help)
    parser.add_argument('--password-policy-disallow-username-substring', required=False, help='Disallow username as a part of the password.', hidden=hidden, **kwargs)