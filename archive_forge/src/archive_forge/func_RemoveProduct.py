from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vision.v1 import vision_v1_messages as messages
def RemoveProduct(self, request, global_params=None):
    """Removes a Product from the specified ProductSet.

      Args:
        request: (VisionProjectsLocationsProductSetsRemoveProductRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('RemoveProduct')
    return self._RunMethod(config, request, global_params=global_params)