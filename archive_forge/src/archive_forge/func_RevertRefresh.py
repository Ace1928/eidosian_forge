from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
def RevertRefresh(self, request, global_params=None):
    """If a call to RefreshWorkspace results in conflicts, use RevertRefresh to.
restore the workspace to the state it was in before the refresh.  Returns
FAILED_PRECONDITION if not preceded by a call to RefreshWorkspace, or if
there are no unresolved conflicts remaining. Returns ABORTED when the
workspace is simultaneously modified by another client.

      Args:
        request: (SourceProjectsReposWorkspacesRevertRefreshRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workspace) The response message.
      """
    config = self.GetMethodConfig('RevertRefresh')
    return self._RunMethod(config, request, global_params=global_params)