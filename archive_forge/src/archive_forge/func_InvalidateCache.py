from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
def InvalidateCache(self, request, global_params=None):
    """Sends a cache invalidation request.

      Args:
        request: (NetworkservicesProjectsLocationsEdgeCacheServicesInvalidateCacheRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InvalidateCacheResponse) The response message.
      """
    config = self.GetMethodConfig('InvalidateCache')
    return self._RunMethod(config, request, global_params=global_params)