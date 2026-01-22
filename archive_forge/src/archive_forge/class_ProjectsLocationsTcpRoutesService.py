from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsTcpRoutesService(base_api.BaseApiService):
    """Service class for the projects_locations_tcpRoutes resource."""
    _NAME = 'projects_locations_tcpRoutes'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsTcpRoutesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new TcpRoute in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsTcpRoutesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tcpRoutes', http_method='POST', method_id='networkservices.projects.locations.tcpRoutes.create', ordered_params=['parent'], path_params=['parent'], query_params=['tcpRouteId'], relative_path='v1/{+parent}/tcpRoutes', request_field='tcpRoute', request_type_name='NetworkservicesProjectsLocationsTcpRoutesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single TcpRoute.

      Args:
        request: (NetworkservicesProjectsLocationsTcpRoutesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tcpRoutes/{tcpRoutesId}', http_method='DELETE', method_id='networkservices.projects.locations.tcpRoutes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsTcpRoutesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single TcpRoute.

      Args:
        request: (NetworkservicesProjectsLocationsTcpRoutesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TcpRoute) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tcpRoutes/{tcpRoutesId}', http_method='GET', method_id='networkservices.projects.locations.tcpRoutes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsTcpRoutesGetRequest', response_type_name='TcpRoute', supports_download=False)

    def List(self, request, global_params=None):
        """Lists TcpRoute in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsTcpRoutesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTcpRoutesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tcpRoutes', http_method='GET', method_id='networkservices.projects.locations.tcpRoutes.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/tcpRoutes', request_field='', request_type_name='NetworkservicesProjectsLocationsTcpRoutesListRequest', response_type_name='ListTcpRoutesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single TcpRoute.

      Args:
        request: (NetworkservicesProjectsLocationsTcpRoutesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tcpRoutes/{tcpRoutesId}', http_method='PATCH', method_id='networkservices.projects.locations.tcpRoutes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='tcpRoute', request_type_name='NetworkservicesProjectsLocationsTcpRoutesPatchRequest', response_type_name='Operation', supports_download=False)