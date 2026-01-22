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
def AddPasswordPolicyPasswordChangeInterval(parser, hidden=False):
    """Add the flag to specify password policy password change interval.

  Args:
    parser: The current argparse parser to add this to.
    hidden: if the field needs to be hidden.
  """
    parser.add_argument('--password-policy-password-change-interval', default=None, type=arg_parsers.Duration(lower_bound='1s'), required=False, help='        Minimum interval after which the password can be changed, for example,\n        2m for 2 minutes. See <a href="/sdk/gcloud/reference/topic/datetimes">\n        $ gcloud topic datetimes</a> for information on duration formats.\n        This flag is available only for PostgreSQL.\n      ', hidden=hidden)