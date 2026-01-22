from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
def ResolveFiles(self, request, global_params=None):
    """Marks files modified as part of a merge as having been resolved. Returns.
ABORTED when the workspace is simultaneously modified by another client.

      Args:
        request: (SourceProjectsReposWorkspacesResolveFilesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workspace) The response message.
      """
    config = self.GetMethodConfig('ResolveFiles')
    return self._RunMethod(config, request, global_params=global_params)