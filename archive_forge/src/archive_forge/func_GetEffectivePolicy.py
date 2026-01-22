from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages as messages
def GetEffectivePolicy(self, request, global_params=None):
    """Gets the effective policy on a resource. This is the result of merging policies in the resource hierarchy and evaluating conditions. The returned policy will not have an `etag` or `condition` set because it is an evaluated policy across multiple resources. Subtrees of Resource Manager resource hierarchy with 'under:' prefix will not be expanded.

      Args:
        request: (OrgpolicyProjectsPoliciesGetEffectivePolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudOrgpolicyV2Policy) The response message.
      """
    config = self.GetMethodConfig('GetEffectivePolicy')
    return self._RunMethod(config, request, global_params=global_params)