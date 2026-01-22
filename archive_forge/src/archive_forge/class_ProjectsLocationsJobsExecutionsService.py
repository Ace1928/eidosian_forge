from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v2 import run_v2_messages as messages
class ProjectsLocationsJobsExecutionsService(base_api.BaseApiService):
    """Service class for the projects_locations_jobs_executions resource."""
    _NAME = 'projects_locations_jobs_executions'

    def __init__(self, client):
        super(RunV2.ProjectsLocationsJobsExecutionsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels an Execution.

      Args:
        request: (RunProjectsLocationsJobsExecutionsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs/{jobsId}/executions/{executionsId}:cancel', http_method='POST', method_id='run.projects.locations.jobs.executions.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:cancel', request_field='googleCloudRunV2CancelExecutionRequest', request_type_name='RunProjectsLocationsJobsExecutionsCancelRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an Execution.

      Args:
        request: (RunProjectsLocationsJobsExecutionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs/{jobsId}/executions/{executionsId}', http_method='DELETE', method_id='run.projects.locations.jobs.executions.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'validateOnly'], relative_path='v2/{+name}', request_field='', request_type_name='RunProjectsLocationsJobsExecutionsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def ExportStatus(self, request, global_params=None):
        """Read the status of an image export operation.

      Args:
        request: (RunProjectsLocationsJobsExecutionsExportStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2ExportStatusResponse) The response message.
      """
        config = self.GetMethodConfig('ExportStatus')
        return self._RunMethod(config, request, global_params=global_params)
    ExportStatus.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs/{jobsId}/executions/{executionsId}/{executionsId1}:exportStatus', http_method='GET', method_id='run.projects.locations.jobs.executions.exportStatus', ordered_params=['name', 'operationId'], path_params=['name', 'operationId'], query_params=[], relative_path='v2/{+name}/{+operationId}:exportStatus', request_field='', request_type_name='RunProjectsLocationsJobsExecutionsExportStatusRequest', response_type_name='GoogleCloudRunV2ExportStatusResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about an Execution.

      Args:
        request: (RunProjectsLocationsJobsExecutionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2Execution) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs/{jobsId}/executions/{executionsId}', http_method='GET', method_id='run.projects.locations.jobs.executions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='RunProjectsLocationsJobsExecutionsGetRequest', response_type_name='GoogleCloudRunV2Execution', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Executions from a Job.

      Args:
        request: (RunProjectsLocationsJobsExecutionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2ListExecutionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/jobs/{jobsId}/executions', http_method='GET', method_id='run.projects.locations.jobs.executions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v2/{+parent}/executions', request_field='', request_type_name='RunProjectsLocationsJobsExecutionsListRequest', response_type_name='GoogleCloudRunV2ListExecutionsResponse', supports_download=False)