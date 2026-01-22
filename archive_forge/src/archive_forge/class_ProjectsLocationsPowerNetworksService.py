from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.marketplacesolutions.v1alpha1 import marketplacesolutions_v1alpha1_messages as messages
class ProjectsLocationsPowerNetworksService(base_api.BaseApiService):
    """Service class for the projects_locations_powerNetworks resource."""
    _NAME = 'projects_locations_powerNetworks'

    def __init__(self, client):
        super(MarketplacesolutionsV1alpha1.ProjectsLocationsPowerNetworksService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get details about a single network from Power.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerNetworksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PowerNetwork) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerNetworks/{powerNetworksId}', http_method='GET', method_id='marketplacesolutions.projects.locations.powerNetworks.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='MarketplacesolutionsProjectsLocationsPowerNetworksGetRequest', response_type_name='PowerNetwork', supports_download=False)

    def List(self, request, global_params=None):
        """List networks in a given project and location from Power.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerNetworksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPowerNetworksResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerNetworks', http_method='GET', method_id='marketplacesolutions.projects.locations.powerNetworks.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/powerNetworks', request_field='', request_type_name='MarketplacesolutionsProjectsLocationsPowerNetworksListRequest', response_type_name='ListPowerNetworksResponse', supports_download=False)