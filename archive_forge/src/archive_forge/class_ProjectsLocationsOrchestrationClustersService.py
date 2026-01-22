from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.telcoautomation.v1 import telcoautomation_v1_messages as messages
class ProjectsLocationsOrchestrationClustersService(base_api.BaseApiService):
    """Service class for the projects_locations_orchestrationClusters resource."""
    _NAME = 'projects_locations_orchestrationClusters'

    def __init__(self, client):
        super(TelcoautomationV1.ProjectsLocationsOrchestrationClustersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new OrchestrationCluster in a given project and location.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters', http_method='POST', method_id='telcoautomation.projects.locations.orchestrationClusters.create', ordered_params=['parent'], path_params=['parent'], query_params=['orchestrationClusterId', 'requestId'], relative_path='v1/{+parent}/orchestrationClusters', request_field='orchestrationCluster', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single OrchestrationCluster.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}', http_method='DELETE', method_id='telcoautomation.projects.locations.orchestrationClusters.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single OrchestrationCluster.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OrchestrationCluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersGetRequest', response_type_name='OrchestrationCluster', supports_download=False)

    def List(self, request, global_params=None):
        """Lists OrchestrationClusters in a given project and location.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOrchestrationClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/orchestrationClusters', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersListRequest', response_type_name='ListOrchestrationClustersResponse', supports_download=False)