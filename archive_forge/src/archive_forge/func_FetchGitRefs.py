from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v2 import cloudbuild_v2_messages as messages
def FetchGitRefs(self, request, global_params=None):
    """Fetch the list of branches or tags for a given repository.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsRepositoriesFetchGitRefsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchGitRefsResponse) The response message.
      """
    config = self.GetMethodConfig('FetchGitRefs')
    return self._RunMethod(config, request, global_params=global_params)