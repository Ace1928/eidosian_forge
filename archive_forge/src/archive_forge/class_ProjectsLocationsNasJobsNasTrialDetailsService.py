from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsNasJobsNasTrialDetailsService(base_api.BaseApiService):
    """Service class for the projects_locations_nasJobs_nasTrialDetails resource."""
    _NAME = 'projects_locations_nasJobs_nasTrialDetails'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsNasJobsNasTrialDetailsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a NasTrialDetail.

      Args:
        request: (AiplatformProjectsLocationsNasJobsNasTrialDetailsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1NasTrialDetail) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/nasJobs/{nasJobsId}/nasTrialDetails/{nasTrialDetailsId}', http_method='GET', method_id='aiplatform.projects.locations.nasJobs.nasTrialDetails.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsNasJobsNasTrialDetailsGetRequest', response_type_name='GoogleCloudAiplatformV1NasTrialDetail', supports_download=False)

    def List(self, request, global_params=None):
        """List top NasTrialDetails of a NasJob.

      Args:
        request: (AiplatformProjectsLocationsNasJobsNasTrialDetailsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListNasTrialDetailsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/nasJobs/{nasJobsId}/nasTrialDetails', http_method='GET', method_id='aiplatform.projects.locations.nasJobs.nasTrialDetails.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/nasTrialDetails', request_field='', request_type_name='AiplatformProjectsLocationsNasJobsNasTrialDetailsListRequest', response_type_name='GoogleCloudAiplatformV1ListNasTrialDetailsResponse', supports_download=False)