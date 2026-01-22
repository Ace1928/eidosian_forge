from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsSchedulesService(base_api.BaseApiService):
    """Service class for the projects_locations_schedules resource."""
    _NAME = 'projects_locations_schedules'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsSchedulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Schedule.

      Args:
        request: (AiplatformProjectsLocationsSchedulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Schedule) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/schedules', http_method='POST', method_id='aiplatform.projects.locations.schedules.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/schedules', request_field='googleCloudAiplatformV1Schedule', request_type_name='AiplatformProjectsLocationsSchedulesCreateRequest', response_type_name='GoogleCloudAiplatformV1Schedule', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Schedule.

      Args:
        request: (AiplatformProjectsLocationsSchedulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/schedules/{schedulesId}', http_method='DELETE', method_id='aiplatform.projects.locations.schedules.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsSchedulesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Schedule.

      Args:
        request: (AiplatformProjectsLocationsSchedulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Schedule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/schedules/{schedulesId}', http_method='GET', method_id='aiplatform.projects.locations.schedules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsSchedulesGetRequest', response_type_name='GoogleCloudAiplatformV1Schedule', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Schedules in a Location.

      Args:
        request: (AiplatformProjectsLocationsSchedulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListSchedulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/schedules', http_method='GET', method_id='aiplatform.projects.locations.schedules.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/schedules', request_field='', request_type_name='AiplatformProjectsLocationsSchedulesListRequest', response_type_name='GoogleCloudAiplatformV1ListSchedulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an active or paused Schedule. When the Schedule is updated, new runs will be scheduled starting from the updated next execution time after the update time based on the time_specification in the updated Schedule. All unstarted runs before the update time will be skipped while already created runs will NOT be paused or canceled.

      Args:
        request: (AiplatformProjectsLocationsSchedulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Schedule) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/schedules/{schedulesId}', http_method='PATCH', method_id='aiplatform.projects.locations.schedules.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1Schedule', request_type_name='AiplatformProjectsLocationsSchedulesPatchRequest', response_type_name='GoogleCloudAiplatformV1Schedule', supports_download=False)

    def Pause(self, request, global_params=None):
        """Pauses a Schedule. Will mark Schedule.state to 'PAUSED'. If the schedule is paused, no new runs will be created. Already created runs will NOT be paused or canceled.

      Args:
        request: (AiplatformProjectsLocationsSchedulesPauseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Pause')
        return self._RunMethod(config, request, global_params=global_params)
    Pause.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/schedules/{schedulesId}:pause', http_method='POST', method_id='aiplatform.projects.locations.schedules.pause', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:pause', request_field='googleCloudAiplatformV1PauseScheduleRequest', request_type_name='AiplatformProjectsLocationsSchedulesPauseRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Resume(self, request, global_params=None):
        """Resumes a paused Schedule to start scheduling new runs. Will mark Schedule.state to 'ACTIVE'. Only paused Schedule can be resumed. When the Schedule is resumed, new runs will be scheduled starting from the next execution time after the current time based on the time_specification in the Schedule. If Schedule.catchUp is set up true, all missed runs will be scheduled for backfill first.

      Args:
        request: (AiplatformProjectsLocationsSchedulesResumeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Resume')
        return self._RunMethod(config, request, global_params=global_params)
    Resume.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/schedules/{schedulesId}:resume', http_method='POST', method_id='aiplatform.projects.locations.schedules.resume', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:resume', request_field='googleCloudAiplatformV1ResumeScheduleRequest', request_type_name='AiplatformProjectsLocationsSchedulesResumeRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)