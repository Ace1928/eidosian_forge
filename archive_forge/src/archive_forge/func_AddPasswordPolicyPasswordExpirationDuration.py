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
def AddPasswordPolicyPasswordExpirationDuration(parser):
    """Add the flag to specify expiration duration after password is updated.

  Args:
    parser: The current argparse parser to add this to.
  """
    parser.add_argument('--password-policy-password-expiration-duration', default=None, type=arg_parsers.Duration(lower_bound='1s'), required=False, help='        Expiration duration after a password is updated, for example,\n        2d for 2 days. See `gcloud topic datetimes` for information on\n        duration formats. This flag is available only for MySQL.\n      ')