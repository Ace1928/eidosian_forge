from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v1 import cloudresourcemanager_v1_messages as messages
def SetOrgPolicy(self, request, global_params=None):
    """Updates the specified `Policy` on the resource. Creates a new `Policy` for that `Constraint` on the resource if one does not exist. Not supplying an `etag` on the request `Policy` results in an unconditional write of the `Policy`.

      Args:
        request: (CloudresourcemanagerProjectsSetOrgPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OrgPolicy) The response message.
      """
    config = self.GetMethodConfig('SetOrgPolicy')
    return self._RunMethod(config, request, global_params=global_params)