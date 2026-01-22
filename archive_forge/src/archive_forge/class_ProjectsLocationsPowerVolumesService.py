from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.marketplacesolutions.v1alpha1 import marketplacesolutions_v1alpha1_messages as messages
class ProjectsLocationsPowerVolumesService(base_api.BaseApiService):
    """Service class for the projects_locations_powerVolumes resource."""
    _NAME = 'projects_locations_powerVolumes'

    def __init__(self, client):
        super(MarketplacesolutionsV1alpha1.ProjectsLocationsPowerVolumesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get details about a single volume from Power.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerVolumesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PowerVolume) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerVolumes/{powerVolumesId}', http_method='GET', method_id='marketplacesolutions.projects.locations.powerVolumes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='MarketplacesolutionsProjectsLocationsPowerVolumesGetRequest', response_type_name='PowerVolume', supports_download=False)

    def List(self, request, global_params=None):
        """List servers in a given project and location from Power.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerVolumesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPowerVolumesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerVolumes', http_method='GET', method_id='marketplacesolutions.projects.locations.powerVolumes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/powerVolumes', request_field='', request_type_name='MarketplacesolutionsProjectsLocationsPowerVolumesListRequest', response_type_name='ListPowerVolumesResponse', supports_download=False)