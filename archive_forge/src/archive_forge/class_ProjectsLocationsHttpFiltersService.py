from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1alpha1 import networkservices_v1alpha1_messages as messages
class ProjectsLocationsHttpFiltersService(base_api.BaseApiService):
    """Service class for the projects_locations_httpFilters resource."""
    _NAME = 'projects_locations_httpFilters'

    def __init__(self, client):
        super(NetworkservicesV1alpha1.ProjectsLocationsHttpFiltersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new HttpFilter in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsHttpFiltersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/httpFilters', http_method='POST', method_id='networkservices.projects.locations.httpFilters.create', ordered_params=['parent'], path_params=['parent'], query_params=['httpFilterId'], relative_path='v1alpha1/{+parent}/httpFilters', request_field='httpFilter', request_type_name='NetworkservicesProjectsLocationsHttpFiltersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single HttpFilter.

      Args:
        request: (NetworkservicesProjectsLocationsHttpFiltersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/httpFilters/{httpFiltersId}', http_method='DELETE', method_id='networkservices.projects.locations.httpFilters.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsHttpFiltersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single HttpFilter.

      Args:
        request: (NetworkservicesProjectsLocationsHttpFiltersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpFilter) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/httpFilters/{httpFiltersId}', http_method='GET', method_id='networkservices.projects.locations.httpFilters.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsHttpFiltersGetRequest', response_type_name='HttpFilter', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworkservicesProjectsLocationsHttpFiltersGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/httpFilters/{httpFiltersId}:getIamPolicy', http_method='GET', method_id='networkservices.projects.locations.httpFilters.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha1/{+resource}:getIamPolicy', request_field='', request_type_name='NetworkservicesProjectsLocationsHttpFiltersGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists HttpFilters in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsHttpFiltersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListHttpFiltersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/httpFilters', http_method='GET', method_id='networkservices.projects.locations.httpFilters.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/httpFilters', request_field='', request_type_name='NetworkservicesProjectsLocationsHttpFiltersListRequest', response_type_name='ListHttpFiltersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single HttpFilter.

      Args:
        request: (NetworkservicesProjectsLocationsHttpFiltersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/httpFilters/{httpFiltersId}', http_method='PATCH', method_id='networkservices.projects.locations.httpFilters.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='httpFilter', request_type_name='NetworkservicesProjectsLocationsHttpFiltersPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworkservicesProjectsLocationsHttpFiltersSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/httpFilters/{httpFiltersId}:setIamPolicy', http_method='POST', method_id='networkservices.projects.locations.httpFilters.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='NetworkservicesProjectsLocationsHttpFiltersSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworkservicesProjectsLocationsHttpFiltersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/httpFilters/{httpFiltersId}:testIamPermissions', http_method='POST', method_id='networkservices.projects.locations.httpFilters.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='NetworkservicesProjectsLocationsHttpFiltersTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)