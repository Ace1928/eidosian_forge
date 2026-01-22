from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v1 import cloudresourcemanager_v1_messages as messages
def GetEffectiveOrgPolicy(self, request, global_params=None):
    """Gets the effective `Policy` on a resource. This is the result of merging `Policies` in the resource hierarchy. The returned `Policy` will not have an `etag`set because it is a computed `Policy` across multiple resources. Subtrees of Resource Manager resource hierarchy with 'under:' prefix will not be expanded.

      Args:
        request: (CloudresourcemanagerProjectsGetEffectiveOrgPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OrgPolicy) The response message.
      """
    config = self.GetMethodConfig('GetEffectiveOrgPolicy')
    return self._RunMethod(config, request, global_params=global_params)