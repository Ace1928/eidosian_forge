from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsNodeTypesService(base_api.BaseApiService):
    """Service class for the projects_locations_nodeTypes resource."""
    _NAME = 'projects_locations_nodeTypes'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsNodeTypesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single `NodeType`.

      Args:
        request: (VmwareengineProjectsLocationsNodeTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/nodeTypes/{nodeTypesId}', http_method='GET', method_id='vmwareengine.projects.locations.nodeTypes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsNodeTypesGetRequest', response_type_name='NodeType', supports_download=False)

    def List(self, request, global_params=None):
        """Lists node types.

      Args:
        request: (VmwareengineProjectsLocationsNodeTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNodeTypesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/nodeTypes', http_method='GET', method_id='vmwareengine.projects.locations.nodeTypes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/nodeTypes', request_field='', request_type_name='VmwareengineProjectsLocationsNodeTypesListRequest', response_type_name='ListNodeTypesResponse', supports_download=False)