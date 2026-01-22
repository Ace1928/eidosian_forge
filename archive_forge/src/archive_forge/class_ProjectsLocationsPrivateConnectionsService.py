from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsPrivateConnectionsService(base_api.BaseApiService):
    """Service class for the projects_locations_privateConnections resource."""
    _NAME = 'projects_locations_privateConnections'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsPrivateConnectionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new private connection that can be used for accessing private Clouds.

      Args:
        request: (VmwareengineProjectsLocationsPrivateConnectionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateConnections', http_method='POST', method_id='vmwareengine.projects.locations.privateConnections.create', ordered_params=['parent'], path_params=['parent'], query_params=['privateConnectionId', 'requestId'], relative_path='v1/{+parent}/privateConnections', request_field='privateConnection', request_type_name='VmwareengineProjectsLocationsPrivateConnectionsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `PrivateConnection` resource. When a private connection is deleted for a VMware Engine network, the connected network becomes inaccessible to that VMware Engine network.

      Args:
        request: (VmwareengineProjectsLocationsPrivateConnectionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateConnections/{privateConnectionsId}', http_method='DELETE', method_id='vmwareengine.projects.locations.privateConnections.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateConnectionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a `PrivateConnection` resource by its resource name. The resource contains details of the private connection, such as connected network, routing mode and state.

      Args:
        request: (VmwareengineProjectsLocationsPrivateConnectionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PrivateConnection) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateConnections/{privateConnectionsId}', http_method='GET', method_id='vmwareengine.projects.locations.privateConnections.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateConnectionsGetRequest', response_type_name='PrivateConnection', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `PrivateConnection` resources in a given project and location.

      Args:
        request: (VmwareengineProjectsLocationsPrivateConnectionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPrivateConnectionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateConnections', http_method='GET', method_id='vmwareengine.projects.locations.privateConnections.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/privateConnections', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateConnectionsListRequest', response_type_name='ListPrivateConnectionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Modifies a `PrivateConnection` resource. Only `description` and `routing_mode` fields can be updated. Only fields specified in `updateMask` are applied.

      Args:
        request: (VmwareengineProjectsLocationsPrivateConnectionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateConnections/{privateConnectionsId}', http_method='PATCH', method_id='vmwareengine.projects.locations.privateConnections.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='privateConnection', request_type_name='VmwareengineProjectsLocationsPrivateConnectionsPatchRequest', response_type_name='Operation', supports_download=False)