from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.connectors.v1 import connectors_v1_messages as messages
class ProjectsLocationsProvidersConnectorsVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_providers_connectors_versions resource."""
    _NAME = 'projects_locations_providers_connectors_versions'

    def __init__(self, client):
        super(ConnectorsV1.ProjectsLocationsProvidersConnectorsVersionsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single connector version.

      Args:
        request: (ConnectorsProjectsLocationsProvidersConnectorsVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConnectorVersion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/providers/{providersId}/connectors/{connectorsId}/versions/{versionsId}', http_method='GET', method_id='connectors.projects.locations.providers.connectors.versions.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='ConnectorsProjectsLocationsProvidersConnectorsVersionsGetRequest', response_type_name='ConnectorVersion', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Connector Versions in a given project and location.

      Args:
        request: (ConnectorsProjectsLocationsProvidersConnectorsVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConnectorVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/providers/{providersId}/connectors/{connectorsId}/versions', http_method='GET', method_id='connectors.projects.locations.providers.connectors.versions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/versions', request_field='', request_type_name='ConnectorsProjectsLocationsProvidersConnectorsVersionsListRequest', response_type_name='ListConnectorVersionsResponse', supports_download=False)