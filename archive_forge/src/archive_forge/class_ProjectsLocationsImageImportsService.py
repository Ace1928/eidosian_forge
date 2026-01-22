from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
class ProjectsLocationsImageImportsService(base_api.BaseApiService):
    """Service class for the projects_locations_imageImports resource."""
    _NAME = 'projects_locations_imageImports'

    def __init__(self, client):
        super(VmmigrationV1.ProjectsLocationsImageImportsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ImageImport in a given project.

      Args:
        request: (VmmigrationProjectsLocationsImageImportsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/imageImports', http_method='POST', method_id='vmmigration.projects.locations.imageImports.create', ordered_params=['parent'], path_params=['parent'], query_params=['imageImportId', 'requestId'], relative_path='v1/{+parent}/imageImports', request_field='imageImport', request_type_name='VmmigrationProjectsLocationsImageImportsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ImageImport.

      Args:
        request: (VmmigrationProjectsLocationsImageImportsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/imageImports/{imageImportsId}', http_method='DELETE', method_id='vmmigration.projects.locations.imageImports.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsImageImportsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ImageImport.

      Args:
        request: (VmmigrationProjectsLocationsImageImportsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ImageImport) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/imageImports/{imageImportsId}', http_method='GET', method_id='vmmigration.projects.locations.imageImports.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsImageImportsGetRequest', response_type_name='ImageImport', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ImageImports in a given project.

      Args:
        request: (VmmigrationProjectsLocationsImageImportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListImageImportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/imageImports', http_method='GET', method_id='vmmigration.projects.locations.imageImports.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/imageImports', request_field='', request_type_name='VmmigrationProjectsLocationsImageImportsListRequest', response_type_name='ListImageImportsResponse', supports_download=False)