from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListPreconfiguredExpressionSets(self, request, global_params=None):
    """Gets the current list of preconfigured Web Application Firewall (WAF) expressions.

      Args:
        request: (ComputeSecurityPoliciesListPreconfiguredExpressionSetsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPoliciesListPreconfiguredExpressionSetsResponse) The response message.
      """
    config = self.GetMethodConfig('ListPreconfiguredExpressionSets')
    return self._RunMethod(config, request, global_params=global_params)