from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListRoutePolicies(self, request, global_params=None):
    """Retrieves a list of router route policy subresources available to the specified project.

      Args:
        request: (ComputeRoutersListRoutePoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RoutersListRoutePolicies) The response message.
      """
    config = self.GetMethodConfig('ListRoutePolicies')
    return self._RunMethod(config, request, global_params=global_params)