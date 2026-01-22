from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appengine.v1beta import appengine_v1beta_messages as messages
def ListRuntimes(self, request, global_params=None):
    """Lists all the available runtimes for the application.

      Args:
        request: (AppengineAppsListRuntimesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRuntimesResponse) The response message.
      """
    config = self.GetMethodConfig('ListRuntimes')
    return self._RunMethod(config, request, global_params=global_params)