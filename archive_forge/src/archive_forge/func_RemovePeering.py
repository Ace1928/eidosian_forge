from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def RemovePeering(self, request, global_params=None):
    """Removes a peering from the specified network.

      Args:
        request: (ComputeNetworksRemovePeeringRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('RemovePeering')
    return self._RunMethod(config, request, global_params=global_params)