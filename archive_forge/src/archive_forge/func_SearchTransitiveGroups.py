from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
def SearchTransitiveGroups(self, request, global_params=None):
    """Search transitive groups of a member. **Note:** This feature is only available to Google Workspace Enterprise Standard, Enterprise Plus, and Enterprise for Education; and Cloud Identity Premium accounts. If the account of the member is not one of these, a 403 (PERMISSION_DENIED) HTTP status code will be returned. A transitive group is any group that has a direct or indirect membership to the member. Actor must have view permissions all transitive groups.

      Args:
        request: (CloudidentityGroupsMembershipsSearchTransitiveGroupsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchTransitiveGroupsResponse) The response message.
      """
    config = self.GetMethodConfig('SearchTransitiveGroups')
    return self._RunMethod(config, request, global_params=global_params)