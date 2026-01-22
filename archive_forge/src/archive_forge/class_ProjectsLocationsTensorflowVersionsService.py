from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.tpu.v1 import tpu_v1_messages as messages
class ProjectsLocationsTensorflowVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_tensorflowVersions resource."""
    _NAME = 'projects_locations_tensorflowVersions'

    def __init__(self, client):
        super(TpuV1.ProjectsLocationsTensorflowVersionsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets TensorFlow Version.

      Args:
        request: (TpuProjectsLocationsTensorflowVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TensorFlowVersion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorflowVersions/{tensorflowVersionsId}', http_method='GET', method_id='tpu.projects.locations.tensorflowVersions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='TpuProjectsLocationsTensorflowVersionsGetRequest', response_type_name='TensorFlowVersion', supports_download=False)

    def List(self, request, global_params=None):
        """List TensorFlow versions supported by this API.

      Args:
        request: (TpuProjectsLocationsTensorflowVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTensorFlowVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorflowVersions', http_method='GET', method_id='tpu.projects.locations.tensorflowVersions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/tensorflowVersions', request_field='', request_type_name='TpuProjectsLocationsTensorflowVersionsListRequest', response_type_name='ListTensorFlowVersionsResponse', supports_download=False)