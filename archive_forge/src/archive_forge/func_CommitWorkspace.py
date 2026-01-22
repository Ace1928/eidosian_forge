from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
def CommitWorkspace(self, request, global_params=None):
    """Commits some or all of the modified files in a workspace. This creates a.
new revision in the repo with the workspace's contents. Returns ABORTED if the workspace ID
in the request contains a snapshot ID and it is not the same as the
workspace's current snapshot ID or if the workspace is simultaneously
modified by another client.

      Args:
        request: (SourceProjectsReposWorkspacesCommitWorkspaceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workspace) The response message.
      """
    config = self.GetMethodConfig('CommitWorkspace')
    return self._RunMethod(config, request, global_params=global_params)