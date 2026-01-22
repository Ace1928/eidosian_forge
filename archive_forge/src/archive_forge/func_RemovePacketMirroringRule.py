from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
def RemovePacketMirroringRule(self, request, global_params=None):
    """Deletes a packet mirroring rule of the specified priority.

      Args:
        request: (ComputeNetworkFirewallPoliciesRemovePacketMirroringRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('RemovePacketMirroringRule')
    return self._RunMethod(config, request, global_params=global_params)