from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def CopyRules(self, request, global_params=None):
    """Copies rules to the specified security policy.

      Args:
        request: (ComputeOrganizationSecurityPoliciesCopyRulesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('CopyRules')
    return self._RunMethod(config, request, global_params=global_params)