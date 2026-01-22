from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsNasJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_nasJobs resource."""
    _NAME = 'projects_locations_nasJobs'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsNasJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels a NasJob. Starts asynchronous cancellation on the NasJob. The server makes a best effort to cancel the job, but success is not guaranteed. Clients can use JobService.GetNasJob or other methods to check whether the cancellation succeeded or whether the job completed despite cancellation. On successful cancellation, the NasJob is not deleted; instead it becomes a job with a NasJob.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`, and NasJob.state is set to `CANCELLED`.

      Args:
        request: (AiplatformProjectsLocationsNasJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/nasJobs/{nasJobsId}:cancel', http_method='POST', method_id='aiplatform.projects.locations.nasJobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='googleCloudAiplatformV1CancelNasJobRequest', request_type_name='AiplatformProjectsLocationsNasJobsCancelRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a NasJob.

      Args:
        request: (AiplatformProjectsLocationsNasJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1NasJob) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/nasJobs', http_method='POST', method_id='aiplatform.projects.locations.nasJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/nasJobs', request_field='googleCloudAiplatformV1NasJob', request_type_name='AiplatformProjectsLocationsNasJobsCreateRequest', response_type_name='GoogleCloudAiplatformV1NasJob', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a NasJob.

      Args:
        request: (AiplatformProjectsLocationsNasJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/nasJobs/{nasJobsId}', http_method='DELETE', method_id='aiplatform.projects.locations.nasJobs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsNasJobsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a NasJob.

      Args:
        request: (AiplatformProjectsLocationsNasJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1NasJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/nasJobs/{nasJobsId}', http_method='GET', method_id='aiplatform.projects.locations.nasJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsNasJobsGetRequest', response_type_name='GoogleCloudAiplatformV1NasJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists NasJobs in a Location.

      Args:
        request: (AiplatformProjectsLocationsNasJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListNasJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/nasJobs', http_method='GET', method_id='aiplatform.projects.locations.nasJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/nasJobs', request_field='', request_type_name='AiplatformProjectsLocationsNasJobsListRequest', response_type_name='GoogleCloudAiplatformV1ListNasJobsResponse', supports_download=False)