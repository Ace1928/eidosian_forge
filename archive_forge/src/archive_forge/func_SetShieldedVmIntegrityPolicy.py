from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetShieldedVmIntegrityPolicy(self, request, global_params=None):
    """Sets the Shielded VM integrity policy for a VM instance. You can only use this method on a running VM instance. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeInstancesSetShieldedVmIntegrityPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetShieldedVmIntegrityPolicy')
    return self._RunMethod(config, request, global_params=global_params)