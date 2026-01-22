from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetEdgeSecurityPolicy(self, request, global_params=None):
    """Sets the edge security policy for the specified backend service.

      Args:
        request: (ComputeBackendServicesSetEdgeSecurityPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetEdgeSecurityPolicy')
    return self._RunMethod(config, request, global_params=global_params)