from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appconfigmanager.v1alpha import appconfigmanager_v1alpha_messages as messages
class ProjectsLocationsConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_configs resource."""
    _NAME = 'projects_locations_configs'

    def __init__(self, client):
        super(AppconfigmanagerV1alpha.ProjectsLocationsConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Config in a given project and location.

      Args:
        request: (AppconfigmanagerProjectsLocationsConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Config) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/configs', http_method='POST', method_id='appconfigmanager.projects.locations.configs.create', ordered_params=['parent'], path_params=['parent'], query_params=['configId', 'requestId'], relative_path='v1alpha/{+parent}/configs', request_field='config', request_type_name='AppconfigmanagerProjectsLocationsConfigsCreateRequest', response_type_name='Config', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Config.

      Args:
        request: (AppconfigmanagerProjectsLocationsConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/configs/{configsId}', http_method='DELETE', method_id='appconfigmanager.projects.locations.configs.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='AppconfigmanagerProjectsLocationsConfigsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Config.

      Args:
        request: (AppconfigmanagerProjectsLocationsConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Config) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/configs/{configsId}', http_method='GET', method_id='appconfigmanager.projects.locations.configs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AppconfigmanagerProjectsLocationsConfigsGetRequest', response_type_name='Config', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Configs in a given project and location.

      Args:
        request: (AppconfigmanagerProjectsLocationsConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/configs', http_method='GET', method_id='appconfigmanager.projects.locations.configs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/configs', request_field='', request_type_name='AppconfigmanagerProjectsLocationsConfigsListRequest', response_type_name='ListConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Config.

      Args:
        request: (AppconfigmanagerProjectsLocationsConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Config) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/configs/{configsId}', http_method='PATCH', method_id='appconfigmanager.projects.locations.configs.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='config', request_type_name='AppconfigmanagerProjectsLocationsConfigsPatchRequest', response_type_name='Config', supports_download=False)