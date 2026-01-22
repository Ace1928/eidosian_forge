from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetSecurityPolicy(self, request, global_params=None):
    """Sets the Google Cloud Armor security policy for the specified target pool. For more information, see Google Cloud Armor Overview.

      Args:
        request: (ComputeTargetPoolsSetSecurityPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetSecurityPolicy')
    return self._RunMethod(config, request, global_params=global_params)