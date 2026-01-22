from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.config.v1alpha2 import config_v1alpha2_messages as messages
class ProjectsLocationsPreviewsService(base_api.BaseApiService):
    """Service class for the projects_locations_previews resource."""
    _NAME = 'projects_locations_previews'

    def __init__(self, client):
        super(ConfigV1alpha2.ProjectsLocationsPreviewsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Preview.

      Args:
        request: (ConfigProjectsLocationsPreviewsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/previews', http_method='POST', method_id='config.projects.locations.previews.create', ordered_params=['parent'], path_params=['parent'], query_params=['previewId', 'requestId'], relative_path='v1alpha2/{+parent}/previews', request_field='preview', request_type_name='ConfigProjectsLocationsPreviewsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Preview.

      Args:
        request: (ConfigProjectsLocationsPreviewsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/previews/{previewsId}', http_method='DELETE', method_id='config.projects.locations.previews.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ConfigProjectsLocationsPreviewsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Export(self, request, global_params=None):
        """Export Preview results.

      Args:
        request: (ConfigProjectsLocationsPreviewsExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExportPreviewResultResponse) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/previews/{previewsId}:export', http_method='POST', method_id='config.projects.locations.previews.export', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}:export', request_field='exportPreviewResultRequest', request_type_name='ConfigProjectsLocationsPreviewsExportRequest', response_type_name='ExportPreviewResultResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details about a Preview.

      Args:
        request: (ConfigProjectsLocationsPreviewsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Preview) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/previews/{previewsId}', http_method='GET', method_id='config.projects.locations.previews.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ConfigProjectsLocationsPreviewsGetRequest', response_type_name='Preview', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Previews in a given project and location.

      Args:
        request: (ConfigProjectsLocationsPreviewsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPreviewsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/previews', http_method='GET', method_id='config.projects.locations.previews.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/previews', request_field='', request_type_name='ConfigProjectsLocationsPreviewsListRequest', response_type_name='ListPreviewsResponse', supports_download=False)