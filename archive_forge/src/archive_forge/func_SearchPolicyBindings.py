from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v3beta import iam_v3beta_messages as messages
def SearchPolicyBindings(self, request, global_params=None):
    """Returns all policy bindings that bind a specific policy if a user has searchPolicyBindings permission on that policy.

      Args:
        request: (IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesSearchPolicyBindingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV3betaSearchPrincipalAccessBoundaryPolicyBindingsResponse) The response message.
      """
    config = self.GetMethodConfig('SearchPolicyBindings')
    return self._RunMethod(config, request, global_params=global_params)