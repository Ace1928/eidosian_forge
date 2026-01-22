from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
class GroupsService(base_api.BaseApiService):
    """Service class for the groups resource."""
    _NAME = 'groups'

    def __init__(self, client):
        super(CloudidentityV1.GroupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Group.

      Args:
        request: (CloudidentityGroupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudidentity.groups.create', ordered_params=[], path_params=[], query_params=['initialGroupConfig'], relative_path='v1/groups', request_field='group', request_type_name='CloudidentityGroupsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `Group`.

      Args:
        request: (CloudidentityGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}', http_method='DELETE', method_id='cloudidentity.groups.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityGroupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a `Group`.

      Args:
        request: (CloudidentityGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Group) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}', http_method='GET', method_id='cloudidentity.groups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityGroupsGetRequest', response_type_name='Group', supports_download=False)

    def GetSecuritySettings(self, request, global_params=None):
        """Get Security Settings.

      Args:
        request: (CloudidentityGroupsGetSecuritySettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecuritySettings) The response message.
      """
        config = self.GetMethodConfig('GetSecuritySettings')
        return self._RunMethod(config, request, global_params=global_params)
    GetSecuritySettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/securitySettings', http_method='GET', method_id='cloudidentity.groups.getSecuritySettings', ordered_params=['name'], path_params=['name'], query_params=['readMask'], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityGroupsGetSecuritySettingsRequest', response_type_name='SecuritySettings', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the `Group` resources under a customer or namespace.

      Args:
        request: (CloudidentityGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudidentity.groups.list', ordered_params=[], path_params=[], query_params=['pageSize', 'pageToken', 'parent', 'view'], relative_path='v1/groups', request_field='', request_type_name='CloudidentityGroupsListRequest', response_type_name='ListGroupsResponse', supports_download=False)

    def Lookup(self, request, global_params=None):
        """Looks up the [resource name](https://cloud.google.com/apis/design/resource_names) of a `Group` by its `EntityKey`.

      Args:
        request: (CloudidentityGroupsLookupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LookupGroupNameResponse) The response message.
      """
        config = self.GetMethodConfig('Lookup')
        return self._RunMethod(config, request, global_params=global_params)
    Lookup.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudidentity.groups.lookup', ordered_params=[], path_params=[], query_params=['groupKey_id', 'groupKey_namespace'], relative_path='v1/groups:lookup', request_field='', request_type_name='CloudidentityGroupsLookupRequest', response_type_name='LookupGroupNameResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a `Group`.

      Args:
        request: (CloudidentityGroupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}', http_method='PATCH', method_id='cloudidentity.groups.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='group', request_type_name='CloudidentityGroupsPatchRequest', response_type_name='Operation', supports_download=False)

    def Search(self, request, global_params=None):
        """Searches for `Group` resources matching a specified query.

      Args:
        request: (CloudidentityGroupsSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudidentity.groups.search', ordered_params=[], path_params=[], query_params=['pageSize', 'pageToken', 'query', 'view'], relative_path='v1/groups:search', request_field='', request_type_name='CloudidentityGroupsSearchRequest', response_type_name='SearchGroupsResponse', supports_download=False)

    def UpdateSecuritySettings(self, request, global_params=None):
        """Update Security Settings.

      Args:
        request: (CloudidentityGroupsUpdateSecuritySettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdateSecuritySettings')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateSecuritySettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/securitySettings', http_method='PATCH', method_id='cloudidentity.groups.updateSecuritySettings', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='securitySettings', request_type_name='CloudidentityGroupsUpdateSecuritySettingsRequest', response_type_name='Operation', supports_download=False)