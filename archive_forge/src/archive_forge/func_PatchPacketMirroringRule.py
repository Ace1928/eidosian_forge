from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
def PatchPacketMirroringRule(self, request, global_params=None):
    """Patches a packet mirroring rule of the specified priority.

      Args:
        request: (ComputeNetworkFirewallPoliciesPatchPacketMirroringRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('PatchPacketMirroringRule')
    return self._RunMethod(config, request, global_params=global_params)