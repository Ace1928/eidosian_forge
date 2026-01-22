from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddActiveDirectoryBackupOperatorsArg(parser):
    """Adds a --backup-operators arg to the given parser."""
    parser.add_argument('--backup-operators', type=arg_parsers.ArgList(element_type=str), metavar='BACKUP_OPERATOR', help='Users to be added to the Built-in Backup Operator Active Directory group.')