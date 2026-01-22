from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v2 import cloudbuild_v2_messages as messages
class ProjectsLocationsTaskRunsService(base_api.BaseApiService):
    """Service class for the projects_locations_taskRuns resource."""
    _NAME = 'projects_locations_taskRuns'

    def __init__(self, client):
        super(CloudbuildV2.ProjectsLocationsTaskRunsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new TaskRun in a given project and location.

      Args:
        request: (CloudbuildProjectsLocationsTaskRunsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/taskRuns', http_method='POST', method_id='cloudbuild.projects.locations.taskRuns.create', ordered_params=['parent'], path_params=['parent'], query_params=['taskRunId', 'validateOnly'], relative_path='v2/{+parent}/taskRuns', request_field='taskRun', request_type_name='CloudbuildProjectsLocationsTaskRunsCreateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single TaskRun.

      Args:
        request: (CloudbuildProjectsLocationsTaskRunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TaskRun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/taskRuns/{taskRunsId}', http_method='GET', method_id='cloudbuild.projects.locations.taskRuns.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsTaskRunsGetRequest', response_type_name='TaskRun', supports_download=False)

    def List(self, request, global_params=None):
        """Lists TaskRuns in a given project and location.

      Args:
        request: (CloudbuildProjectsLocationsTaskRunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTaskRunsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/taskRuns', http_method='GET', method_id='cloudbuild.projects.locations.taskRuns.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/taskRuns', request_field='', request_type_name='CloudbuildProjectsLocationsTaskRunsListRequest', response_type_name='ListTaskRunsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single TaskRun.

      Args:
        request: (CloudbuildProjectsLocationsTaskRunsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/taskRuns/{taskRunsId}', http_method='PATCH', method_id='cloudbuild.projects.locations.taskRuns.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask', 'validateOnly'], relative_path='v2/{+name}', request_field='taskRun', request_type_name='CloudbuildProjectsLocationsTaskRunsPatchRequest', response_type_name='Operation', supports_download=False)