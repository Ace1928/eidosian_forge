from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def PatchRule(self, request, global_params=None):
    """Patches a rule at the specified priority. To clear fields in the rule, leave the fields empty and specify them in the updateMask.

      Args:
        request: (ComputeSecurityPoliciesPatchRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('PatchRule')
    return self._RunMethod(config, request, global_params=global_params)