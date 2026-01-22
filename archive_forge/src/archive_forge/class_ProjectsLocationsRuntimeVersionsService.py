from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.tpu.v2 import tpu_v2_messages as messages
class ProjectsLocationsRuntimeVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_runtimeVersions resource."""
    _NAME = 'projects_locations_runtimeVersions'

    def __init__(self, client):
        super(TpuV2.ProjectsLocationsRuntimeVersionsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a runtime version.

      Args:
        request: (TpuProjectsLocationsRuntimeVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RuntimeVersion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/runtimeVersions/{runtimeVersionsId}', http_method='GET', method_id='tpu.projects.locations.runtimeVersions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='TpuProjectsLocationsRuntimeVersionsGetRequest', response_type_name='RuntimeVersion', supports_download=False)

    def List(self, request, global_params=None):
        """Lists runtime versions supported by this API.

      Args:
        request: (TpuProjectsLocationsRuntimeVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRuntimeVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/runtimeVersions', http_method='GET', method_id='tpu.projects.locations.runtimeVersions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/runtimeVersions', request_field='', request_type_name='TpuProjectsLocationsRuntimeVersionsListRequest', response_type_name='ListRuntimeVersionsResponse', supports_download=False)