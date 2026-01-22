from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
class ProjectsLocationsJobsStagesService(base_api.BaseApiService):
    """Service class for the projects_locations_jobs_stages resource."""
    _NAME = 'projects_locations_jobs_stages'

    def __init__(self, client):
        super(DataflowV1b3.ProjectsLocationsJobsStagesService, self).__init__(client)
        self._upload_configs = {}

    def GetExecutionDetails(self, request, global_params=None):
        """Request detailed information about the execution status of a stage of the job. EXPERIMENTAL. This API is subject to change or removal without notice.

      Args:
        request: (DataflowProjectsLocationsJobsStagesGetExecutionDetailsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StageExecutionDetails) The response message.
      """
        config = self.GetMethodConfig('GetExecutionDetails')
        return self._RunMethod(config, request, global_params=global_params)
    GetExecutionDetails.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataflow.projects.locations.jobs.stages.getExecutionDetails', ordered_params=['projectId', 'location', 'jobId', 'stageId'], path_params=['jobId', 'location', 'projectId', 'stageId'], query_params=['endTime', 'pageSize', 'pageToken', 'startTime'], relative_path='v1b3/projects/{projectId}/locations/{location}/jobs/{jobId}/stages/{stageId}/executionDetails', request_field='', request_type_name='DataflowProjectsLocationsJobsStagesGetExecutionDetailsRequest', response_type_name='StageExecutionDetails', supports_download=False)