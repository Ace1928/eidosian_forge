from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsModelDeploymentMonitoringJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_modelDeploymentMonitoringJobs resource."""
    _NAME = 'projects_locations_modelDeploymentMonitoringJobs'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsModelDeploymentMonitoringJobsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a ModelDeploymentMonitoringJob. It will run periodically on a configured interval.

      Args:
        request: (AiplatformProjectsLocationsModelDeploymentMonitoringJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ModelDeploymentMonitoringJob) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/modelDeploymentMonitoringJobs', http_method='POST', method_id='aiplatform.projects.locations.modelDeploymentMonitoringJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/modelDeploymentMonitoringJobs', request_field='googleCloudAiplatformV1ModelDeploymentMonitoringJob', request_type_name='AiplatformProjectsLocationsModelDeploymentMonitoringJobsCreateRequest', response_type_name='GoogleCloudAiplatformV1ModelDeploymentMonitoringJob', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a ModelDeploymentMonitoringJob.

      Args:
        request: (AiplatformProjectsLocationsModelDeploymentMonitoringJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/modelDeploymentMonitoringJobs/{modelDeploymentMonitoringJobsId}', http_method='DELETE', method_id='aiplatform.projects.locations.modelDeploymentMonitoringJobs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsModelDeploymentMonitoringJobsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a ModelDeploymentMonitoringJob.

      Args:
        request: (AiplatformProjectsLocationsModelDeploymentMonitoringJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ModelDeploymentMonitoringJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/modelDeploymentMonitoringJobs/{modelDeploymentMonitoringJobsId}', http_method='GET', method_id='aiplatform.projects.locations.modelDeploymentMonitoringJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsModelDeploymentMonitoringJobsGetRequest', response_type_name='GoogleCloudAiplatformV1ModelDeploymentMonitoringJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ModelDeploymentMonitoringJobs in a Location.

      Args:
        request: (AiplatformProjectsLocationsModelDeploymentMonitoringJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListModelDeploymentMonitoringJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/modelDeploymentMonitoringJobs', http_method='GET', method_id='aiplatform.projects.locations.modelDeploymentMonitoringJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/modelDeploymentMonitoringJobs', request_field='', request_type_name='AiplatformProjectsLocationsModelDeploymentMonitoringJobsListRequest', response_type_name='GoogleCloudAiplatformV1ListModelDeploymentMonitoringJobsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a ModelDeploymentMonitoringJob.

      Args:
        request: (AiplatformProjectsLocationsModelDeploymentMonitoringJobsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/modelDeploymentMonitoringJobs/{modelDeploymentMonitoringJobsId}', http_method='PATCH', method_id='aiplatform.projects.locations.modelDeploymentMonitoringJobs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1ModelDeploymentMonitoringJob', request_type_name='AiplatformProjectsLocationsModelDeploymentMonitoringJobsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Pause(self, request, global_params=None):
        """Pauses a ModelDeploymentMonitoringJob. If the job is running, the server makes a best effort to cancel the job. Will mark ModelDeploymentMonitoringJob.state to 'PAUSED'.

      Args:
        request: (AiplatformProjectsLocationsModelDeploymentMonitoringJobsPauseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Pause')
        return self._RunMethod(config, request, global_params=global_params)
    Pause.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/modelDeploymentMonitoringJobs/{modelDeploymentMonitoringJobsId}:pause', http_method='POST', method_id='aiplatform.projects.locations.modelDeploymentMonitoringJobs.pause', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:pause', request_field='googleCloudAiplatformV1PauseModelDeploymentMonitoringJobRequest', request_type_name='AiplatformProjectsLocationsModelDeploymentMonitoringJobsPauseRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Resume(self, request, global_params=None):
        """Resumes a paused ModelDeploymentMonitoringJob. It will start to run from next scheduled time. A deleted ModelDeploymentMonitoringJob can't be resumed.

      Args:
        request: (AiplatformProjectsLocationsModelDeploymentMonitoringJobsResumeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Resume')
        return self._RunMethod(config, request, global_params=global_params)
    Resume.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/modelDeploymentMonitoringJobs/{modelDeploymentMonitoringJobsId}:resume', http_method='POST', method_id='aiplatform.projects.locations.modelDeploymentMonitoringJobs.resume', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:resume', request_field='googleCloudAiplatformV1ResumeModelDeploymentMonitoringJobRequest', request_type_name='AiplatformProjectsLocationsModelDeploymentMonitoringJobsResumeRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def SearchModelDeploymentMonitoringStatsAnomalies(self, request, global_params=None):
        """Searches Model Monitoring Statistics generated within a given time window.

      Args:
        request: (AiplatformProjectsLocationsModelDeploymentMonitoringJobsSearchModelDeploymentMonitoringStatsAnomaliesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1SearchModelDeploymentMonitoringStatsAnomaliesResponse) The response message.
      """
        config = self.GetMethodConfig('SearchModelDeploymentMonitoringStatsAnomalies')
        return self._RunMethod(config, request, global_params=global_params)
    SearchModelDeploymentMonitoringStatsAnomalies.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/modelDeploymentMonitoringJobs/{modelDeploymentMonitoringJobsId}:searchModelDeploymentMonitoringStatsAnomalies', http_method='POST', method_id='aiplatform.projects.locations.modelDeploymentMonitoringJobs.searchModelDeploymentMonitoringStatsAnomalies', ordered_params=['modelDeploymentMonitoringJob'], path_params=['modelDeploymentMonitoringJob'], query_params=[], relative_path='v1/{+modelDeploymentMonitoringJob}:searchModelDeploymentMonitoringStatsAnomalies', request_field='googleCloudAiplatformV1SearchModelDeploymentMonitoringStatsAnomaliesRequest', request_type_name='AiplatformProjectsLocationsModelDeploymentMonitoringJobsSearchModelDeploymentMonitoringStatsAnomaliesRequest', response_type_name='GoogleCloudAiplatformV1SearchModelDeploymentMonitoringStatsAnomaliesResponse', supports_download=False)