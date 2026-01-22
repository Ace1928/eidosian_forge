from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
class ProjectsLocationsSessionTemplatesService(base_api.BaseApiService):
    """Service class for the projects_locations_sessionTemplates resource."""
    _NAME = 'projects_locations_sessionTemplates'

    def __init__(self, client):
        super(DataprocV1.ProjectsLocationsSessionTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a session template synchronously.

      Args:
        request: (DataprocProjectsLocationsSessionTemplatesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SessionTemplate) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessionTemplates', http_method='POST', method_id='dataproc.projects.locations.sessionTemplates.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/sessionTemplates', request_field='sessionTemplate', request_type_name='DataprocProjectsLocationsSessionTemplatesCreateRequest', response_type_name='SessionTemplate', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a session template.

      Args:
        request: (DataprocProjectsLocationsSessionTemplatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessionTemplates/{sessionTemplatesId}', http_method='DELETE', method_id='dataproc.projects.locations.sessionTemplates.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataprocProjectsLocationsSessionTemplatesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the resource representation for a session template.

      Args:
        request: (DataprocProjectsLocationsSessionTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SessionTemplate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessionTemplates/{sessionTemplatesId}', http_method='GET', method_id='dataproc.projects.locations.sessionTemplates.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataprocProjectsLocationsSessionTemplatesGetRequest', response_type_name='SessionTemplate', supports_download=False)

    def List(self, request, global_params=None):
        """Lists session templates.

      Args:
        request: (DataprocProjectsLocationsSessionTemplatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSessionTemplatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessionTemplates', http_method='GET', method_id='dataproc.projects.locations.sessionTemplates.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/sessionTemplates', request_field='', request_type_name='DataprocProjectsLocationsSessionTemplatesListRequest', response_type_name='ListSessionTemplatesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the session template synchronously.

      Args:
        request: (SessionTemplate) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SessionTemplate) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessionTemplates/{sessionTemplatesId}', http_method='PATCH', method_id='dataproc.projects.locations.sessionTemplates.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='SessionTemplate', response_type_name='SessionTemplate', supports_download=False)