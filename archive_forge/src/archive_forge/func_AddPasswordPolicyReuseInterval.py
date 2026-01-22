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
def AddPasswordPolicyReuseInterval(parser, hidden=False):
    """Add the flag to specify password policy reuse interval.

  Args:
    parser: The current argparse parser to add this to.
    hidden: if the field needs to be hidden.
  """
    parser.add_argument('--password-policy-reuse-interval', type=arg_parsers.BoundedInt(lower_bound=0, upper_bound=100), required=False, default=None, help='Number of previous passwords that cannot be reused. The valid range is 0 to 100.', hidden=hidden)