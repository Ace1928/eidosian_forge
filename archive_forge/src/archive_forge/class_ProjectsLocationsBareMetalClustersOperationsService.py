from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
class ProjectsLocationsBareMetalClustersOperationsService(base_api.BaseApiService):
    """Service class for the projects_locations_bareMetalClusters_operations resource."""
    _NAME = 'projects_locations_bareMetalClusters_operations'

    def __init__(self, client):
        super(GkeonpremV1.ProjectsLocationsBareMetalClustersOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}/operations/{operationsId}', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalClusters.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersOperationsGetRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns `UNIMPLEMENTED`.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}/operations', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalClusters.operations.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+name}/operations', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersOperationsListRequest', response_type_name='ListOperationsResponse', supports_download=False)