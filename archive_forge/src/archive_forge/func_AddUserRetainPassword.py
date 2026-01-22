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
def AddUserRetainPassword(parser):
    """Will retain the old password when changing to the new password.

  Args:
    parser: The current argparse parser to add this to.
  """
    kwargs = _GetKwargsForBoolFlag(False)
    parser.add_argument('--retain-password', required=False, help='Retain the old password when changing to the new password. Must set password with this flag. This flag is only available for MySQL 8.0.', **kwargs)