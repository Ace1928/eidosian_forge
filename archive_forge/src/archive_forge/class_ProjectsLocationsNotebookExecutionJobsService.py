from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class ProjectsLocationsNotebookExecutionJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_notebookExecutionJobs resource."""
    _NAME = 'projects_locations_notebookExecutionJobs'

    def __init__(self, client):
        super(AiplatformV1beta1.ProjectsLocationsNotebookExecutionJobsService, self).__init__(client)
        self._upload_configs = {}

    def GenerateAccessToken(self, request, global_params=None):
        """Internal only: Called from Compute Engine instance to obtain EUC for owner Anonymous access: authenticates caller using VM identity JWT. Design doc: go/colab-on-vertex-euc-dd.

      Args:
        request: (AiplatformProjectsLocationsNotebookExecutionJobsGenerateAccessTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1GenerateAccessTokenResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateAccessToken')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateAccessToken.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/notebookExecutionJobs/{notebookExecutionJobsId}:generateAccessToken', http_method='POST', method_id='aiplatform.projects.locations.notebookExecutionJobs.generateAccessToken', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}:generateAccessToken', request_field='googleCloudAiplatformV1beta1GenerateAccessTokenRequest', request_type_name='AiplatformProjectsLocationsNotebookExecutionJobsGenerateAccessTokenRequest', response_type_name='GoogleCloudAiplatformV1beta1GenerateAccessTokenResponse', supports_download=False)

    def ReportEvent(self, request, global_params=None):
        """ReportEvent method for the projects_locations_notebookExecutionJobs service.

      Args:
        request: (AiplatformProjectsLocationsNotebookExecutionJobsReportEventRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1ReportExecutionEventResponse) The response message.
      """
        config = self.GetMethodConfig('ReportEvent')
        return self._RunMethod(config, request, global_params=global_params)
    ReportEvent.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/notebookExecutionJobs/{notebookExecutionJobsId}:reportEvent', http_method='POST', method_id='aiplatform.projects.locations.notebookExecutionJobs.reportEvent', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}:reportEvent', request_field='googleCloudAiplatformV1beta1ReportExecutionEventRequest', request_type_name='AiplatformProjectsLocationsNotebookExecutionJobsReportEventRequest', response_type_name='GoogleCloudAiplatformV1beta1ReportExecutionEventResponse', supports_download=False)