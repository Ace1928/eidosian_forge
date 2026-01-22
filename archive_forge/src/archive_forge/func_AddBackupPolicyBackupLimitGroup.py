from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupPolicyBackupLimitGroup(parser):
    """Adds a parser argument group for backup limits.

    Flags include:
    --daily-backup-limit
    --weekly-backup-limit
    --monthly-backup-limit

  Args:
    parser: The argparser.
  """
    backup_limit_group = parser.add_group(help='Add backup limit arguments.')
    AddBackupPolicyDailyBackupLimitArg(backup_limit_group)
    AddBackupPolicyWeeklyBackupLimitArg(backup_limit_group)
    AddBackupPolicyMonthlyBackupLimitArg(backup_limit_group)