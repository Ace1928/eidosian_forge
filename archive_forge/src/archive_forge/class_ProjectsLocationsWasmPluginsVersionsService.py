from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsWasmPluginsVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_wasmPlugins_versions resource."""
    _NAME = 'projects_locations_wasmPlugins_versions'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsWasmPluginsVersionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `WasmPluginVersion` resource in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsWasmPluginsVersionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/wasmPlugins/{wasmPluginsId}/versions', http_method='POST', method_id='networkservices.projects.locations.wasmPlugins.versions.create', ordered_params=['parent'], path_params=['parent'], query_params=['wasmPluginVersionId'], relative_path='v1/{+parent}/versions', request_field='wasmPluginVersion', request_type_name='NetworkservicesProjectsLocationsWasmPluginsVersionsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified `WasmPluginVersion` resource.

      Args:
        request: (NetworkservicesProjectsLocationsWasmPluginsVersionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/wasmPlugins/{wasmPluginsId}/versions/{versionsId}', http_method='DELETE', method_id='networkservices.projects.locations.wasmPlugins.versions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsWasmPluginsVersionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of the specified `WasmPluginVersion` resource.

      Args:
        request: (NetworkservicesProjectsLocationsWasmPluginsVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WasmPluginVersion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/wasmPlugins/{wasmPluginsId}/versions/{versionsId}', http_method='GET', method_id='networkservices.projects.locations.wasmPlugins.versions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsWasmPluginsVersionsGetRequest', response_type_name='WasmPluginVersion', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `WasmPluginVersion` resources in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsWasmPluginsVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWasmPluginVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/wasmPlugins/{wasmPluginsId}/versions', http_method='GET', method_id='networkservices.projects.locations.wasmPlugins.versions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/versions', request_field='', request_type_name='NetworkservicesProjectsLocationsWasmPluginsVersionsListRequest', response_type_name='ListWasmPluginVersionsResponse', supports_download=False)