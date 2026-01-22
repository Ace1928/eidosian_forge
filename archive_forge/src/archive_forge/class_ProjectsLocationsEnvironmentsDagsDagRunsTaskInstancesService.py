from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
class ProjectsLocationsEnvironmentsDagsDagRunsTaskInstancesService(base_api.BaseApiService):
    """Service class for the projects_locations_environments_dags_dagRuns_taskInstances resource."""
    _NAME = 'projects_locations_environments_dags_dagRuns_taskInstances'

    def __init__(self, client):
        super(ComposerV1alpha2.ProjectsLocationsEnvironmentsDagsDagRunsTaskInstancesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves a task instance.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsDagRunsTaskInstancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TaskInstance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/dags/{dagsId}/dagRuns/{dagRunsId}/taskInstances/{taskInstancesId}', http_method='GET', method_id='composer.projects.locations.environments.dags.dagRuns.taskInstances.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsDagsDagRunsTaskInstancesGetRequest', response_type_name='TaskInstance', supports_download=False)

    def List(self, request, global_params=None):
        """Lists task instances for a specified DAG run.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsDagsDagRunsTaskInstancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTaskInstancesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/dags/{dagsId}/dagRuns/{dagRunsId}/taskInstances', http_method='GET', method_id='composer.projects.locations.environments.dags.dagRuns.taskInstances.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/taskInstances', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsDagsDagRunsTaskInstancesListRequest', response_type_name='ListTaskInstancesResponse', supports_download=False)