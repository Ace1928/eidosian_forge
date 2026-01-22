from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v2 import cloudbuild_v2_messages as messages
class ProjectsLocationsPipelineRunsService(base_api.BaseApiService):
    """Service class for the projects_locations_pipelineRuns resource."""
    _NAME = 'projects_locations_pipelineRuns'

    def __init__(self, client):
        super(CloudbuildV2.ProjectsLocationsPipelineRunsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new PipelineRun in a given project and location.

      Args:
        request: (CloudbuildProjectsLocationsPipelineRunsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/pipelineRuns', http_method='POST', method_id='cloudbuild.projects.locations.pipelineRuns.create', ordered_params=['parent'], path_params=['parent'], query_params=['pipelineRunId', 'validateOnly'], relative_path='v2/{+parent}/pipelineRuns', request_field='pipelineRun', request_type_name='CloudbuildProjectsLocationsPipelineRunsCreateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single PipelineRun.

      Args:
        request: (CloudbuildProjectsLocationsPipelineRunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PipelineRun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/pipelineRuns/{pipelineRunsId}', http_method='GET', method_id='cloudbuild.projects.locations.pipelineRuns.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsPipelineRunsGetRequest', response_type_name='PipelineRun', supports_download=False)

    def List(self, request, global_params=None):
        """Lists PipelineRuns in a given project and location.

      Args:
        request: (CloudbuildProjectsLocationsPipelineRunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPipelineRunsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/pipelineRuns', http_method='GET', method_id='cloudbuild.projects.locations.pipelineRuns.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/pipelineRuns', request_field='', request_type_name='CloudbuildProjectsLocationsPipelineRunsListRequest', response_type_name='ListPipelineRunsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single PipelineRun.

      Args:
        request: (CloudbuildProjectsLocationsPipelineRunsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/pipelineRuns/{pipelineRunsId}', http_method='PATCH', method_id='cloudbuild.projects.locations.pipelineRuns.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask', 'validateOnly'], relative_path='v2/{+name}', request_field='pipelineRun', request_type_name='CloudbuildProjectsLocationsPipelineRunsPatchRequest', response_type_name='Operation', supports_download=False)