from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.source.v1 import source_v1_messages as messages
def ReadFromWorkspaceOrAlias(self, request, global_params=None):
    """ReadFromWorkspaceOrAlias performs a Read using either the most recent.
snapshot of the given workspace, if the workspace exists, or the
revision referred to by the given alias if the workspace does not exist.

      Args:
        request: (SourceProjectsReposFilesReadFromWorkspaceOrAliasRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReadResponse) The response message.
      """
    config = self.GetMethodConfig('ReadFromWorkspaceOrAlias')
    return self._RunMethod(config, request, global_params=global_params)