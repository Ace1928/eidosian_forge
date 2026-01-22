from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsHttpRoutesService(base_api.BaseApiService):
    """Service class for the projects_locations_httpRoutes resource."""
    _NAME = 'projects_locations_httpRoutes'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsHttpRoutesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new HttpRoute in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsHttpRoutesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/httpRoutes', http_method='POST', method_id='networkservices.projects.locations.httpRoutes.create', ordered_params=['parent'], path_params=['parent'], query_params=['httpRouteId'], relative_path='v1/{+parent}/httpRoutes', request_field='httpRoute', request_type_name='NetworkservicesProjectsLocationsHttpRoutesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single HttpRoute.

      Args:
        request: (NetworkservicesProjectsLocationsHttpRoutesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/httpRoutes/{httpRoutesId}', http_method='DELETE', method_id='networkservices.projects.locations.httpRoutes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsHttpRoutesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single HttpRoute.

      Args:
        request: (NetworkservicesProjectsLocationsHttpRoutesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpRoute) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/httpRoutes/{httpRoutesId}', http_method='GET', method_id='networkservices.projects.locations.httpRoutes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsHttpRoutesGetRequest', response_type_name='HttpRoute', supports_download=False)

    def List(self, request, global_params=None):
        """Lists HttpRoute in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsHttpRoutesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListHttpRoutesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/httpRoutes', http_method='GET', method_id='networkservices.projects.locations.httpRoutes.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/httpRoutes', request_field='', request_type_name='NetworkservicesProjectsLocationsHttpRoutesListRequest', response_type_name='ListHttpRoutesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single HttpRoute.

      Args:
        request: (NetworkservicesProjectsLocationsHttpRoutesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/httpRoutes/{httpRoutesId}', http_method='PATCH', method_id='networkservices.projects.locations.httpRoutes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='httpRoute', request_type_name='NetworkservicesProjectsLocationsHttpRoutesPatchRequest', response_type_name='Operation', supports_download=False)