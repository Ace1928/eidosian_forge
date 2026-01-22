from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
def Reindex(self, request, global_params=None):
    """Updates the index files for an OS repository. Intended for use on remote repositories to check if the upstream has been updated, and if so pull the new index files.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesReindexRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Reindex')
    return self._RunMethod(config, request, global_params=global_params)