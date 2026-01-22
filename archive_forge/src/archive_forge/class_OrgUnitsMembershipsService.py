from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1beta1 import cloudidentity_v1beta1_messages as messages
class OrgUnitsMembershipsService(base_api.BaseApiService):
    """Service class for the orgUnits_memberships resource."""
    _NAME = 'orgUnits_memberships'

    def __init__(self, client):
        super(CloudidentityV1beta1.OrgUnitsMembershipsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List OrgMembership resources in an OrgUnit treated as 'parent'. Parent format: orgUnits/{$orgUnitId} where `$orgUnitId` is the `orgUnitId` from the [Admin SDK `OrgUnit` resource](https://developers.google.com/admin-sdk/directory/reference/rest/v1/orgunits).

      Args:
        request: (CloudidentityOrgUnitsMembershipsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOrgMembershipsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/orgUnits/{orgUnitsId}/memberships', http_method='GET', method_id='cloudidentity.orgUnits.memberships.list', ordered_params=['parent'], path_params=['parent'], query_params=['customer', 'filter', 'pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/memberships', request_field='', request_type_name='CloudidentityOrgUnitsMembershipsListRequest', response_type_name='ListOrgMembershipsResponse', supports_download=False)

    def Move(self, request, global_params=None):
        """Move an OrgMembership to a new OrgUnit. NOTE: This is an atomic copy-and-delete. The resource will have a new copy under the destination OrgUnit and be deleted from the source OrgUnit. The resource can only be searched under the destination OrgUnit afterwards.

      Args:
        request: (CloudidentityOrgUnitsMembershipsMoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Move')
        return self._RunMethod(config, request, global_params=global_params)
    Move.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/orgUnits/{orgUnitsId}/memberships/{membershipsId}:move', http_method='POST', method_id='cloudidentity.orgUnits.memberships.move', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}:move', request_field='moveOrgMembershipRequest', request_type_name='CloudidentityOrgUnitsMembershipsMoveRequest', response_type_name='Operation', supports_download=False)