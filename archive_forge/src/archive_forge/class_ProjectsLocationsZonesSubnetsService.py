from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.edgenetwork.v1 import edgenetwork_v1_messages as messages
class ProjectsLocationsZonesSubnetsService(base_api.BaseApiService):
    """Service class for the projects_locations_zones_subnets resource."""
    _NAME = 'projects_locations_zones_subnets'

    def __init__(self, client):
        super(EdgenetworkV1.ProjectsLocationsZonesSubnetsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Subnet in a given project and location.

      Args:
        request: (EdgenetworkProjectsLocationsZonesSubnetsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/subnets', http_method='POST', method_id='edgenetwork.projects.locations.zones.subnets.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'subnetId'], relative_path='v1/{+parent}/subnets', request_field='subnet', request_type_name='EdgenetworkProjectsLocationsZonesSubnetsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Subnet.

      Args:
        request: (EdgenetworkProjectsLocationsZonesSubnetsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/subnets/{subnetsId}', http_method='DELETE', method_id='edgenetwork.projects.locations.zones.subnets.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesSubnetsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Subnet.

      Args:
        request: (EdgenetworkProjectsLocationsZonesSubnetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Subnet) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/subnets/{subnetsId}', http_method='GET', method_id='edgenetwork.projects.locations.zones.subnets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesSubnetsGetRequest', response_type_name='Subnet', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Subnets in a given project and location.

      Args:
        request: (EdgenetworkProjectsLocationsZonesSubnetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSubnetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/subnets', http_method='GET', method_id='edgenetwork.projects.locations.zones.subnets.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/subnets', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesSubnetsListRequest', response_type_name='ListSubnetsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Subnet.

      Args:
        request: (EdgenetworkProjectsLocationsZonesSubnetsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/subnets/{subnetsId}', http_method='PATCH', method_id='edgenetwork.projects.locations.zones.subnets.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='subnet', request_type_name='EdgenetworkProjectsLocationsZonesSubnetsPatchRequest', response_type_name='Operation', supports_download=False)