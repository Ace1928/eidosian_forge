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
def AddBackupName(parser):
    """Add the flag for the NAME of the backup.

  Args:
    parser: The current parser to add this argument.
  """
    parser.add_argument('name', help='The NAME of the backup. To find the NAME, run the following command: $ gcloud sql backups list --project-level.')