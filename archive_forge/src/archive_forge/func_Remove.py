from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
def Remove(self, request, global_params=None):
    """Service producers can use this method to remove private DNS zones in the shared producer host project and matching peering zones in the consumer project.

      Args:
        request: (ServicenetworkingServicesDnsZonesRemoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Remove')
    return self._RunMethod(config, request, global_params=global_params)