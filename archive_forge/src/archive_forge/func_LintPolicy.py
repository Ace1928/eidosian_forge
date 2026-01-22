from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
def LintPolicy(self, request, global_params=None):
    """Lints, or validates, an IAM policy. Currently checks the google.iam.v1.Binding.condition field, which contains a condition expression for a role binding. Successful calls to this method always return an HTTP `200 OK` status code, even if the linter detects an issue in the IAM policy.

      Args:
        request: (LintPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LintPolicyResponse) The response message.
      """
    config = self.GetMethodConfig('LintPolicy')
    return self._RunMethod(config, request, global_params=global_params)