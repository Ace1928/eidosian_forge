from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.edgecontainer.v1beta import edgecontainer_v1beta_messages as messages
class ProjectsLocationsVpnConnectionsService(base_api.BaseApiService):
    """Service class for the projects_locations_vpnConnections resource."""
    _NAME = 'projects_locations_vpnConnections'

    def __init__(self, client):
        super(EdgecontainerV1beta.ProjectsLocationsVpnConnectionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new VPN connection in a given project and location.

      Args:
        request: (EdgecontainerProjectsLocationsVpnConnectionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/vpnConnections', http_method='POST', method_id='edgecontainer.projects.locations.vpnConnections.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'vpnConnectionId'], relative_path='v1beta/{+parent}/vpnConnections', request_field='vpnConnection', request_type_name='EdgecontainerProjectsLocationsVpnConnectionsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single VPN connection.

      Args:
        request: (EdgecontainerProjectsLocationsVpnConnectionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/vpnConnections/{vpnConnectionsId}', http_method='DELETE', method_id='edgecontainer.projects.locations.vpnConnections.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1beta/{+name}', request_field='', request_type_name='EdgecontainerProjectsLocationsVpnConnectionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single VPN connection.

      Args:
        request: (EdgecontainerProjectsLocationsVpnConnectionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VpnConnection) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/vpnConnections/{vpnConnectionsId}', http_method='GET', method_id='edgecontainer.projects.locations.vpnConnections.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='EdgecontainerProjectsLocationsVpnConnectionsGetRequest', response_type_name='VpnConnection', supports_download=False)

    def List(self, request, global_params=None):
        """Lists VPN connections in a given project and location.

      Args:
        request: (EdgecontainerProjectsLocationsVpnConnectionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListVpnConnectionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/vpnConnections', http_method='GET', method_id='edgecontainer.projects.locations.vpnConnections.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/vpnConnections', request_field='', request_type_name='EdgecontainerProjectsLocationsVpnConnectionsListRequest', response_type_name='ListVpnConnectionsResponse', supports_download=False)