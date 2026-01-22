from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appconfigmanager.v1alpha import appconfigmanager_v1alpha_messages as messages
class ProjectsLocationsConfigsVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_configs_versions resource."""
    _NAME = 'projects_locations_configs_versions'

    def __init__(self, client):
        super(AppconfigmanagerV1alpha.ProjectsLocationsConfigsVersionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ConfigVersion in a given project, location, and Config.

      Args:
        request: (AppconfigmanagerProjectsLocationsConfigsVersionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigVersion) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/configs/{configsId}/versions', http_method='POST', method_id='appconfigmanager.projects.locations.configs.versions.create', ordered_params=['parent'], path_params=['parent'], query_params=['configVersionId', 'requestId'], relative_path='v1alpha/{+parent}/versions', request_field='configVersion', request_type_name='AppconfigmanagerProjectsLocationsConfigsVersionsCreateRequest', response_type_name='ConfigVersion', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ConfigVersion.

      Args:
        request: (AppconfigmanagerProjectsLocationsConfigsVersionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/configs/{configsId}/versions/{versionsId}', http_method='DELETE', method_id='appconfigmanager.projects.locations.configs.versions.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='AppconfigmanagerProjectsLocationsConfigsVersionsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ConfigVersion.

      Args:
        request: (AppconfigmanagerProjectsLocationsConfigsVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigVersion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/configs/{configsId}/versions/{versionsId}', http_method='GET', method_id='appconfigmanager.projects.locations.configs.versions.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1alpha/{+name}', request_field='', request_type_name='AppconfigmanagerProjectsLocationsConfigsVersionsGetRequest', response_type_name='ConfigVersion', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ConfigVersions in a given project, location, and Config.

      Args:
        request: (AppconfigmanagerProjectsLocationsConfigsVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConfigVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/configs/{configsId}/versions', http_method='GET', method_id='appconfigmanager.projects.locations.configs.versions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'view'], relative_path='v1alpha/{+parent}/versions', request_field='', request_type_name='AppconfigmanagerProjectsLocationsConfigsVersionsListRequest', response_type_name='ListConfigVersionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single ConfigVersion.

      Args:
        request: (AppconfigmanagerProjectsLocationsConfigsVersionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigVersion) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/configs/{configsId}/versions/{versionsId}', http_method='PATCH', method_id='appconfigmanager.projects.locations.configs.versions.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='configVersion', request_type_name='AppconfigmanagerProjectsLocationsConfigsVersionsPatchRequest', response_type_name='ConfigVersion', supports_download=False)