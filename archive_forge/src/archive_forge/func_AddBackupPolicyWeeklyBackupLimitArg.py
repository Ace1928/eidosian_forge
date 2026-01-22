from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddBackupPolicyWeeklyBackupLimitArg(backup_limit_group):
    """Adds a --weekly-backup-limit arg to the given parser argument group."""
    backup_limit_group.add_argument('--weekly-backup-limit', type=arg_parsers.BoundedInt(lower_bound=0, upper_bound=sys.maxsize), help='\n          Number of weekly backups to keep.\n          Note that the sum of daily, weekly and monthly backups\n          should be greater than 1\n          ')