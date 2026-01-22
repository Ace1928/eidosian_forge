from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
import six
def _ComplementConversionWorkspaceInfo(self, conversion_workspace, args):
    """Returns the conversion workspace info with the supplied or the latest commit id.

    Args:
      conversion_workspace: the internal migration job conversion workspace
        message.
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Raises:
      BadArgumentException: Unable to fetch latest commit for the specified
      conversion workspace.
      InvalidArgumentException: Invalid conversion workspace message on the
      migration job.
    """
    if conversion_workspace.name is None:
        raise exceptions.InvalidArgumentException('conversion-workspace', 'The supplied migration job does not have a valid conversion workspace attached to it')
    if args.commit_id is not None:
        conversion_workspace.commitId = args.commit_id
        return conversion_workspace
    cw_client = conversion_workspaces.ConversionWorkspacesClient(self.release_track)
    cst_conversion_workspace = cw_client.Describe(conversion_workspace.name)
    if cst_conversion_workspace.latestCommitId is None:
        raise exceptions.BadArgumentException('conversion-workspace', 'Unable to fetch latest commit for the specified conversion workspace. Conversion Workspace might not be committed.')
    conversion_workspace.commitId = cst_conversion_workspace.latestCommitId
    return conversion_workspace