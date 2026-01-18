from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v3beta import iam_v3beta_messages as messages
Returns policies (along with the bindings that bind them) that apply to the specified target_query. This means the policies that are bound to the target or any of its ancestors. target_query can be a principal, a principalSet or in the future a resource.

      Args:
        request: (IamSearchApplicablePoliciesSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV3betaSearchApplicablePoliciesResponse) The response message.
      