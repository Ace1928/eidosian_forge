from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.edgenetwork.v1 import edgenetwork_v1_messages as messages
class ProjectsLocationsZonesService(base_api.BaseApiService):
    """Service class for the projects_locations_zones resource."""
    _NAME = 'projects_locations_zones'

    def __init__(self, client):
        super(EdgenetworkV1.ProjectsLocationsZonesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Deprecated: not implemented. Gets details of a single Zone.

      Args:
        request: (EdgenetworkProjectsLocationsZonesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Zone) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}', http_method='GET', method_id='edgenetwork.projects.locations.zones.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesGetRequest', response_type_name='Zone', supports_download=False)

    def Initialize(self, request, global_params=None):
        """InitializeZone will initialize resources for a zone in a project.

      Args:
        request: (EdgenetworkProjectsLocationsZonesInitializeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InitializeZoneResponse) The response message.
      """
        config = self.GetMethodConfig('Initialize')
        return self._RunMethod(config, request, global_params=global_params)
    Initialize.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}:initialize', http_method='POST', method_id='edgenetwork.projects.locations.zones.initialize', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:initialize', request_field='initializeZoneRequest', request_type_name='EdgenetworkProjectsLocationsZonesInitializeRequest', response_type_name='InitializeZoneResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Deprecated: not implemented. Lists Zones in a given project and location.

      Args:
        request: (EdgenetworkProjectsLocationsZonesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListZonesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones', http_method='GET', method_id='edgenetwork.projects.locations.zones.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/zones', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesListRequest', response_type_name='ListZonesResponse', supports_download=False)