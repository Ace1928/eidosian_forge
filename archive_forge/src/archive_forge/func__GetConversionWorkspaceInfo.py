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
def _GetConversionWorkspaceInfo(self, conversion_workspace_ref, args):
    """Returns the conversion worksapce info.

    Args:
      conversion_workspace_ref: str, the reference of the conversion workspace.
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Raises:
      BadArgumentException: Unable to fetch latest commit for the specified
      conversion workspace.
    """
    if conversion_workspace_ref is not None:
        conversion_workspace_obj = self.messages.ConversionWorkspaceInfo(name=conversion_workspace_ref.RelativeName())
        if args.commit_id is not None:
            conversion_workspace_obj.commitId = args.commit_id
        else:
            cw_client = conversion_workspaces.ConversionWorkspacesClient(self.release_track)
            conversion_workspace = cw_client.Describe(conversion_workspace_ref.RelativeName())
            if conversion_workspace.latestCommitId is None:
                raise exceptions.BadArgumentException('conversion-workspace', 'Unable to fetch latest commit for the specified conversion workspace. Conversion Workspace might not be committed.')
            conversion_workspace_obj.commitId = conversion_workspace.latestCommitId
        return conversion_workspace_obj