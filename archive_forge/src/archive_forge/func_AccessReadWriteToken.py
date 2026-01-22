from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v2 import cloudbuild_v2_messages as messages
def AccessReadWriteToken(self, request, global_params=None):
    """Fetches read/write token of a given repository.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsRepositoriesAccessReadWriteTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchReadWriteTokenResponse) The response message.
      """
    config = self.GetMethodConfig('AccessReadWriteToken')
    return self._RunMethod(config, request, global_params=global_params)