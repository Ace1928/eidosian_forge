from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
def DeleteNamedSet(self, request, global_params=None):
    """Deletes Named Set.

      Args:
        request: (ComputeRoutersDeleteNamedSetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DeleteNamedSet')
    return self._RunMethod(config, request, global_params=global_params)