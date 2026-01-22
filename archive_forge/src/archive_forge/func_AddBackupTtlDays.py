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
def AddBackupTtlDays(parser):
    """Add the flag to specify the retention days of the backup.

  Args:
    parser: The current parser to add this argument.
  """
    parser.add_argument('--ttl-days', required=False, type=arg_parsers.BoundedInt(1, 365, unlimited=False), default=None, hidden=True, help=' Specifies the number of days to retain the final backup. The valid range is between 1 and 365. The Default value is 30 days. Provide either ttl-days or expiry-time.')