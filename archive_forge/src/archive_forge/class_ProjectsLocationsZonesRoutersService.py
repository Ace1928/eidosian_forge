from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.edgenetwork.v1 import edgenetwork_v1_messages as messages
class ProjectsLocationsZonesRoutersService(base_api.BaseApiService):
    """Service class for the projects_locations_zones_routers resource."""
    _NAME = 'projects_locations_zones_routers'

    def __init__(self, client):
        super(EdgenetworkV1.ProjectsLocationsZonesRoutersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Router in a given project and location.

      Args:
        request: (EdgenetworkProjectsLocationsZonesRoutersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/routers', http_method='POST', method_id='edgenetwork.projects.locations.zones.routers.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'routerId'], relative_path='v1/{+parent}/routers', request_field='router', request_type_name='EdgenetworkProjectsLocationsZonesRoutersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Router.

      Args:
        request: (EdgenetworkProjectsLocationsZonesRoutersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/routers/{routersId}', http_method='DELETE', method_id='edgenetwork.projects.locations.zones.routers.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesRoutersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Diagnose(self, request, global_params=None):
        """Get the diagnostics of a single router resource.

      Args:
        request: (EdgenetworkProjectsLocationsZonesRoutersDiagnoseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiagnoseRouterResponse) The response message.
      """
        config = self.GetMethodConfig('Diagnose')
        return self._RunMethod(config, request, global_params=global_params)
    Diagnose.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/routers/{routersId}:diagnose', http_method='GET', method_id='edgenetwork.projects.locations.zones.routers.diagnose', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:diagnose', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesRoutersDiagnoseRequest', response_type_name='DiagnoseRouterResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Router.

      Args:
        request: (EdgenetworkProjectsLocationsZonesRoutersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Router) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/routers/{routersId}', http_method='GET', method_id='edgenetwork.projects.locations.zones.routers.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesRoutersGetRequest', response_type_name='Router', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Routers in a given project and location.

      Args:
        request: (EdgenetworkProjectsLocationsZonesRoutersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRoutersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/routers', http_method='GET', method_id='edgenetwork.projects.locations.zones.routers.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/routers', request_field='', request_type_name='EdgenetworkProjectsLocationsZonesRoutersListRequest', response_type_name='ListRoutersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Router.

      Args:
        request: (EdgenetworkProjectsLocationsZonesRoutersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/zones/{zonesId}/routers/{routersId}', http_method='PATCH', method_id='edgenetwork.projects.locations.zones.routers.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='router', request_type_name='EdgenetworkProjectsLocationsZonesRoutersPatchRequest', response_type_name='Operation', supports_download=False)