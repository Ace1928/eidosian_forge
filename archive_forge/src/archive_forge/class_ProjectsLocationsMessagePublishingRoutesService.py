from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1beta1 import networkservices_v1beta1_messages as messages
class ProjectsLocationsMessagePublishingRoutesService(base_api.BaseApiService):
    """Service class for the projects_locations_messagePublishingRoutes resource."""
    _NAME = 'projects_locations_messagePublishingRoutes'

    def __init__(self, client):
        super(NetworkservicesV1beta1.ProjectsLocationsMessagePublishingRoutesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new MessagePublishingRoute in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMessagePublishingRoutesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/messagePublishingRoutes', http_method='POST', method_id='networkservices.projects.locations.messagePublishingRoutes.create', ordered_params=['parent'], path_params=['parent'], query_params=['messagePublishingRouteId'], relative_path='v1beta1/{+parent}/messagePublishingRoutes', request_field='messagePublishingRoute', request_type_name='NetworkservicesProjectsLocationsMessagePublishingRoutesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single MessagePublishingRoute.

      Args:
        request: (NetworkservicesProjectsLocationsMessagePublishingRoutesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/messagePublishingRoutes/{messagePublishingRoutesId}', http_method='DELETE', method_id='networkservices.projects.locations.messagePublishingRoutes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMessagePublishingRoutesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single MessagePublishingRoute.

      Args:
        request: (NetworkservicesProjectsLocationsMessagePublishingRoutesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MessagePublishingRoute) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/messagePublishingRoutes/{messagePublishingRoutesId}', http_method='GET', method_id='networkservices.projects.locations.messagePublishingRoutes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMessagePublishingRoutesGetRequest', response_type_name='MessagePublishingRoute', supports_download=False)

    def List(self, request, global_params=None):
        """Lists MessagePublishingRoute in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMessagePublishingRoutesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMessagePublishingRoutesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/messagePublishingRoutes', http_method='GET', method_id='networkservices.projects.locations.messagePublishingRoutes.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/messagePublishingRoutes', request_field='', request_type_name='NetworkservicesProjectsLocationsMessagePublishingRoutesListRequest', response_type_name='ListMessagePublishingRoutesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single MessagePublishingRoute.

      Args:
        request: (NetworkservicesProjectsLocationsMessagePublishingRoutesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/messagePublishingRoutes/{messagePublishingRoutesId}', http_method='PATCH', method_id='networkservices.projects.locations.messagePublishingRoutes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta1/{+name}', request_field='messagePublishingRoute', request_type_name='NetworkservicesProjectsLocationsMessagePublishingRoutesPatchRequest', response_type_name='Operation', supports_download=False)