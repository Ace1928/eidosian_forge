from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def UpdateRoutePolicy(self, request, global_params=None):
    """Updates or creates new Route Policy.

      Args:
        request: (ComputeRoutersUpdateRoutePolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpdateRoutePolicy')
    return self._RunMethod(config, request, global_params=global_params)