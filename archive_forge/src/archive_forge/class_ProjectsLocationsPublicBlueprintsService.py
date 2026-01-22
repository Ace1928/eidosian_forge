from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.telcoautomation.v1 import telcoautomation_v1_messages as messages
class ProjectsLocationsPublicBlueprintsService(base_api.BaseApiService):
    """Service class for the projects_locations_publicBlueprints resource."""
    _NAME = 'projects_locations_publicBlueprints'

    def __init__(self, client):
        super(TelcoautomationV1.ProjectsLocationsPublicBlueprintsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns the requested public blueprint.

      Args:
        request: (TelcoautomationProjectsLocationsPublicBlueprintsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PublicBlueprint) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/publicBlueprints/{publicBlueprintsId}', http_method='GET', method_id='telcoautomation.projects.locations.publicBlueprints.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='TelcoautomationProjectsLocationsPublicBlueprintsGetRequest', response_type_name='PublicBlueprint', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the blueprints in TNA's public catalog. Default page size = 20, Max Page Size = 100.

      Args:
        request: (TelcoautomationProjectsLocationsPublicBlueprintsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPublicBlueprintsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/publicBlueprints', http_method='GET', method_id='telcoautomation.projects.locations.publicBlueprints.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/publicBlueprints', request_field='', request_type_name='TelcoautomationProjectsLocationsPublicBlueprintsListRequest', response_type_name='ListPublicBlueprintsResponse', supports_download=False)