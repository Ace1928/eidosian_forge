from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def UpdateShieldedVmConfig(self, request, global_params=None):
    """Updates the Shielded VM config for a VM instance. You can only use this method on a stopped VM instance. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeInstancesUpdateShieldedVmConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpdateShieldedVmConfig')
    return self._RunMethod(config, request, global_params=global_params)