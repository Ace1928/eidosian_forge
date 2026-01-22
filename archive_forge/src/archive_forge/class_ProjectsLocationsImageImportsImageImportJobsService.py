from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
class ProjectsLocationsImageImportsImageImportJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_imageImports_imageImportJobs resource."""
    _NAME = 'projects_locations_imageImports_imageImportJobs'

    def __init__(self, client):
        super(VmmigrationV1.ProjectsLocationsImageImportsImageImportJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Initiates the cancellation of a running clone job.

      Args:
        request: (VmmigrationProjectsLocationsImageImportsImageImportJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/imageImports/{imageImportsId}/imageImportJobs/{imageImportJobsId}:cancel', http_method='POST', method_id='vmmigration.projects.locations.imageImports.imageImportJobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='cancelImageImportJobRequest', request_type_name='VmmigrationProjectsLocationsImageImportsImageImportJobsCancelRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ImageImportJob.

      Args:
        request: (VmmigrationProjectsLocationsImageImportsImageImportJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ImageImportJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/imageImports/{imageImportsId}/imageImportJobs/{imageImportJobsId}', http_method='GET', method_id='vmmigration.projects.locations.imageImports.imageImportJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsImageImportsImageImportJobsGetRequest', response_type_name='ImageImportJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ImageImportJobs in a given project.

      Args:
        request: (VmmigrationProjectsLocationsImageImportsImageImportJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListImageImportJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/imageImports/{imageImportsId}/imageImportJobs', http_method='GET', method_id='vmmigration.projects.locations.imageImports.imageImportJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/imageImportJobs', request_field='', request_type_name='VmmigrationProjectsLocationsImageImportsImageImportJobsListRequest', response_type_name='ListImageImportJobsResponse', supports_download=False)