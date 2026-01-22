from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1beta1 import networksecurity_v1beta1_messages as messages
class ProjectsLocationsSecurityProfilesService(base_api.BaseApiService):
    """Service class for the projects_locations_securityProfiles resource."""
    _NAME = 'projects_locations_securityProfiles'

    def __init__(self, client):
        super(NetworksecurityV1beta1.ProjectsLocationsSecurityProfilesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new SecurityProfile in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsSecurityProfilesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/securityProfiles', http_method='POST', method_id='networksecurity.projects.locations.securityProfiles.create', ordered_params=['parent'], path_params=['parent'], query_params=['securityProfileId'], relative_path='v1beta1/{+parent}/securityProfiles', request_field='securityProfile', request_type_name='NetworksecurityProjectsLocationsSecurityProfilesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single SecurityProfile.

      Args:
        request: (NetworksecurityProjectsLocationsSecurityProfilesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/securityProfiles/{securityProfilesId}', http_method='DELETE', method_id='networksecurity.projects.locations.securityProfiles.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1beta1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsSecurityProfilesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single SecurityProfile.

      Args:
        request: (NetworksecurityProjectsLocationsSecurityProfilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityProfile) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/securityProfiles/{securityProfilesId}', http_method='GET', method_id='networksecurity.projects.locations.securityProfiles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsSecurityProfilesGetRequest', response_type_name='SecurityProfile', supports_download=False)

    def List(self, request, global_params=None):
        """Lists SecurityProfiles in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsSecurityProfilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSecurityProfilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/securityProfiles', http_method='GET', method_id='networksecurity.projects.locations.securityProfiles.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/securityProfiles', request_field='', request_type_name='NetworksecurityProjectsLocationsSecurityProfilesListRequest', response_type_name='ListSecurityProfilesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single SecurityProfile.

      Args:
        request: (NetworksecurityProjectsLocationsSecurityProfilesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/securityProfiles/{securityProfilesId}', http_method='PATCH', method_id='networksecurity.projects.locations.securityProfiles.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta1/{+name}', request_field='securityProfile', request_type_name='NetworksecurityProjectsLocationsSecurityProfilesPatchRequest', response_type_name='Operation', supports_download=False)