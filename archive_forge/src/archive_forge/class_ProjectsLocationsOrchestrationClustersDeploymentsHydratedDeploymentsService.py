from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.telcoautomation.v1 import telcoautomation_v1_messages as messages
class ProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsService(base_api.BaseApiService):
    """Service class for the projects_locations_orchestrationClusters_deployments_hydratedDeployments resource."""
    _NAME = 'projects_locations_orchestrationClusters_deployments_hydratedDeployments'

    def __init__(self, client):
        super(TelcoautomationV1.ProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def Apply(self, request, global_params=None):
        """Applies a hydrated deployment to a workload cluster.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsApplyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HydratedDeployment) The response message.
      """
        config = self.GetMethodConfig('Apply')
        return self._RunMethod(config, request, global_params=global_params)
    Apply.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments/{deploymentsId}/hydratedDeployments/{hydratedDeploymentsId}:apply', http_method='POST', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.hydratedDeployments.apply', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:apply', request_field='applyHydratedDeploymentRequest', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsApplyRequest', response_type_name='HydratedDeployment', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the requested hydrated deployment.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HydratedDeployment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments/{deploymentsId}/hydratedDeployments/{hydratedDeploymentsId}', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.hydratedDeployments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsGetRequest', response_type_name='HydratedDeployment', supports_download=False)

    def List(self, request, global_params=None):
        """List all hydrated deployments present under a deployment.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListHydratedDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments/{deploymentsId}/hydratedDeployments', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.hydratedDeployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/hydratedDeployments', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsListRequest', response_type_name='ListHydratedDeploymentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a hydrated deployment.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HydratedDeployment) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments/{deploymentsId}/hydratedDeployments/{hydratedDeploymentsId}', http_method='PATCH', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.hydratedDeployments.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='hydratedDeployment', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsHydratedDeploymentsPatchRequest', response_type_name='HydratedDeployment', supports_download=False)