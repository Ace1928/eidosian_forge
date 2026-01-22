from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datastream.v1 import datastream_v1_messages as messages
class ProjectsLocationsPrivateConnectionsRoutesService(base_api.BaseApiService):
    """Service class for the projects_locations_privateConnections_routes resource."""
    _NAME = 'projects_locations_privateConnections_routes'

    def __init__(self, client):
        super(DatastreamV1.ProjectsLocationsPrivateConnectionsRoutesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Use this method to create a route for a private connectivity configuration in a project and location.

      Args:
        request: (DatastreamProjectsLocationsPrivateConnectionsRoutesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateConnections/{privateConnectionsId}/routes', http_method='POST', method_id='datastream.projects.locations.privateConnections.routes.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'routeId'], relative_path='v1/{+parent}/routes', request_field='route', request_type_name='DatastreamProjectsLocationsPrivateConnectionsRoutesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Use this method to delete a route.

      Args:
        request: (DatastreamProjectsLocationsPrivateConnectionsRoutesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateConnections/{privateConnectionsId}/routes/{routesId}', http_method='DELETE', method_id='datastream.projects.locations.privateConnections.routes.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='DatastreamProjectsLocationsPrivateConnectionsRoutesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Use this method to get details about a route.

      Args:
        request: (DatastreamProjectsLocationsPrivateConnectionsRoutesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Route) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateConnections/{privateConnectionsId}/routes/{routesId}', http_method='GET', method_id='datastream.projects.locations.privateConnections.routes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatastreamProjectsLocationsPrivateConnectionsRoutesGetRequest', response_type_name='Route', supports_download=False)

    def List(self, request, global_params=None):
        """Use this method to list routes created for a private connectivity configuration in a project and location.

      Args:
        request: (DatastreamProjectsLocationsPrivateConnectionsRoutesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRoutesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateConnections/{privateConnectionsId}/routes', http_method='GET', method_id='datastream.projects.locations.privateConnections.routes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/routes', request_field='', request_type_name='DatastreamProjectsLocationsPrivateConnectionsRoutesListRequest', response_type_name='ListRoutesResponse', supports_download=False)