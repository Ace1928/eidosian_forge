from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
def SearchTransitiveMemberships(self, request, global_params=None):
    """Search transitive memberships of a group. **Note:** This feature is only available to Google Workspace Enterprise Standard, Enterprise Plus, and Enterprise for Education; and Cloud Identity Premium accounts. If the account of the group is not one of these, a 403 (PERMISSION_DENIED) HTTP status code will be returned. A transitive membership is any direct or indirect membership of a group. Actor must have view permissions to all transitive memberships.

      Args:
        request: (CloudidentityGroupsMembershipsSearchTransitiveMembershipsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchTransitiveMembershipsResponse) The response message.
      """
    config = self.GetMethodConfig('SearchTransitiveMemberships')
    return self._RunMethod(config, request, global_params=global_params)