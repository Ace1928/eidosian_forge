from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v1 import cloudresourcemanager_v1_messages as messages
def ListOrgPolicies(self, request, global_params=None):
    """Lists all the `Policies` set for a particular resource.

      Args:
        request: (CloudresourcemanagerProjectsListOrgPoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOrgPoliciesResponse) The response message.
      """
    config = self.GetMethodConfig('ListOrgPolicies')
    return self._RunMethod(config, request, global_params=global_params)