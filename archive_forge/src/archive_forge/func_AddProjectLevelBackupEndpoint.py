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
def AddProjectLevelBackupEndpoint(parser):
    """Add the flag to specify requests to route to new backup service end point.

  Args:
    parser: The current argparse parser to add this to.
  """
    parser.add_argument('--project-level', hidden=True, required=False, default=False, action='store_true', help="If true, then invoke project level backup endpoint. Use 'Name' as the value for backup ID. You can find the 'Name' by running $ gcloud sql backups list --project-level.")