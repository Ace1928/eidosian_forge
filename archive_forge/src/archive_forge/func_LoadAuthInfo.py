from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
def LoadAuthInfo(self, request, global_params=None):
    """Load auth info for a server.

      Args:
        request: (BaremetalsolutionProjectsLocationsInstancesLoadAuthInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LoadInstanceAuthInfoResponse) The response message.
      """
    config = self.GetMethodConfig('LoadAuthInfo')
    return self._RunMethod(config, request, global_params=global_params)