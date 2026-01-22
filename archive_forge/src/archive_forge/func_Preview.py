from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def Preview(self, request, global_params=None):
    """Preview fields auto-generated during router create and update operations. Calling this method does NOT create or update the router.

      Args:
        request: (ComputeRoutersPreviewRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RoutersPreviewResponse) The response message.
      """
    config = self.GetMethodConfig('Preview')
    return self._RunMethod(config, request, global_params=global_params)