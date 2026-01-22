from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupPolicyDailyBackupLimitArg(backup_limit_group):
    """Adds a --daily-backup-limit arg to the given parser argument group."""
    backup_limit_group.add_argument('--daily-backup-limit', type=arg_parsers.BoundedInt(lower_bound=MIN_DAILY_BACKUP_LIMIT, upper_bound=sys.maxsize), help='\n          Maximum number of daily backups to keep.\n          Note that the minimum daily backup limit is 2.\n          ')