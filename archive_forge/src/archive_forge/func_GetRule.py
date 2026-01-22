from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetRule(self, request, global_params=None):
    """Gets a rule at the specified priority.

      Args:
        request: (ComputeSecurityPoliciesGetRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPolicyRule) The response message.
      """
    config = self.GetMethodConfig('GetRule')
    return self._RunMethod(config, request, global_params=global_params)