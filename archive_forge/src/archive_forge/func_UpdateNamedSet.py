from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
def UpdateNamedSet(self, request, global_params=None):
    """Updates or creates new Named Set.

      Args:
        request: (ComputeRoutersUpdateNamedSetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpdateNamedSet')
    return self._RunMethod(config, request, global_params=global_params)