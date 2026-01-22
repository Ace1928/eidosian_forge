from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.discovery.v1 import discovery_v1_messages as messages
def GetRest(self, request, global_params=None):
    """Retrieve the description of a particular version of an api.

      Args:
        request: (DiscoveryApisGetRestRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RestDescription) The response message.
      """
    config = self.GetMethodConfig('GetRest')
    return self._RunMethod(config, request, global_params=global_params)