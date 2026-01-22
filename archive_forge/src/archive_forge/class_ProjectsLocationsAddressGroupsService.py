from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
class ProjectsLocationsAddressGroupsService(base_api.BaseApiService):
    """Service class for the projects_locations_addressGroups resource."""
    _NAME = 'projects_locations_addressGroups'

    def __init__(self, client):
        super(NetworksecurityV1.ProjectsLocationsAddressGroupsService, self).__init__(client)
        self._upload_configs = {}

    def AddItems(self, request, global_params=None):
        """Adds items to an address group.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsAddItemsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddItems')
        return self._RunMethod(config, request, global_params=global_params)
    AddItems.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/addressGroups/{addressGroupsId}:addItems', http_method='POST', method_id='networksecurity.projects.locations.addressGroups.addItems', ordered_params=['addressGroup'], path_params=['addressGroup'], query_params=[], relative_path='v1/{+addressGroup}:addItems', request_field='addAddressGroupItemsRequest', request_type_name='NetworksecurityProjectsLocationsAddressGroupsAddItemsRequest', response_type_name='Operation', supports_download=False)

    def CloneItems(self, request, global_params=None):
        """Clones items from one address group to another.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsCloneItemsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CloneItems')
        return self._RunMethod(config, request, global_params=global_params)
    CloneItems.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/addressGroups/{addressGroupsId}:cloneItems', http_method='POST', method_id='networksecurity.projects.locations.addressGroups.cloneItems', ordered_params=['addressGroup'], path_params=['addressGroup'], query_params=[], relative_path='v1/{+addressGroup}:cloneItems', request_field='cloneAddressGroupItemsRequest', request_type_name='NetworksecurityProjectsLocationsAddressGroupsCloneItemsRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new address group in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/addressGroups', http_method='POST', method_id='networksecurity.projects.locations.addressGroups.create', ordered_params=['parent'], path_params=['parent'], query_params=['addressGroupId', 'requestId'], relative_path='v1/{+parent}/addressGroups', request_field='addressGroup', request_type_name='NetworksecurityProjectsLocationsAddressGroupsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single address group.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/addressGroups/{addressGroupsId}', http_method='DELETE', method_id='networksecurity.projects.locations.addressGroups.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsAddressGroupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single address group.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AddressGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/addressGroups/{addressGroupsId}', http_method='GET', method_id='networksecurity.projects.locations.addressGroups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsAddressGroupsGetRequest', response_type_name='AddressGroup', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/addressGroups/{addressGroupsId}:getIamPolicy', http_method='GET', method_id='networksecurity.projects.locations.addressGroups.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='NetworksecurityProjectsLocationsAddressGroupsGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists address groups in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAddressGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/addressGroups', http_method='GET', method_id='networksecurity.projects.locations.addressGroups.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/addressGroups', request_field='', request_type_name='NetworksecurityProjectsLocationsAddressGroupsListRequest', response_type_name='ListAddressGroupsResponse', supports_download=False)

    def ListReferences(self, request, global_params=None):
        """Lists references of an address group.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsListReferencesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAddressGroupReferencesResponse) The response message.
      """
        config = self.GetMethodConfig('ListReferences')
        return self._RunMethod(config, request, global_params=global_params)
    ListReferences.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/addressGroups/{addressGroupsId}:listReferences', http_method='GET', method_id='networksecurity.projects.locations.addressGroups.listReferences', ordered_params=['addressGroup'], path_params=['addressGroup'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+addressGroup}:listReferences', request_field='', request_type_name='NetworksecurityProjectsLocationsAddressGroupsListReferencesRequest', response_type_name='ListAddressGroupReferencesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single address group.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/addressGroups/{addressGroupsId}', http_method='PATCH', method_id='networksecurity.projects.locations.addressGroups.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='addressGroup', request_type_name='NetworksecurityProjectsLocationsAddressGroupsPatchRequest', response_type_name='Operation', supports_download=False)

    def RemoveItems(self, request, global_params=None):
        """Removes items from an address group.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsRemoveItemsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveItems')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveItems.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/addressGroups/{addressGroupsId}:removeItems', http_method='POST', method_id='networksecurity.projects.locations.addressGroups.removeItems', ordered_params=['addressGroup'], path_params=['addressGroup'], query_params=[], relative_path='v1/{+addressGroup}:removeItems', request_field='removeAddressGroupItemsRequest', request_type_name='NetworksecurityProjectsLocationsAddressGroupsRemoveItemsRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/addressGroups/{addressGroupsId}:setIamPolicy', http_method='POST', method_id='networksecurity.projects.locations.addressGroups.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='NetworksecurityProjectsLocationsAddressGroupsSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/addressGroups/{addressGroupsId}:testIamPermissions', http_method='POST', method_id='networksecurity.projects.locations.addressGroups.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='NetworksecurityProjectsLocationsAddressGroupsTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)