from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class ProjectsLocationsConfigurationsService(base_api.BaseApiService):
    """Service class for the projects_locations_configurations resource."""
    _NAME = 'projects_locations_configurations'

    def __init__(self, client):
        super(RunV1.ProjectsLocationsConfigurationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get information about a configuration.

      Args:
        request: (RunProjectsLocationsConfigurationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Configuration) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/configurations/{configurationsId}', http_method='GET', method_id='run.projects.locations.configurations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='RunProjectsLocationsConfigurationsGetRequest', response_type_name='Configuration', supports_download=False)

    def List(self, request, global_params=None):
        """List configurations.

      Args:
        request: (RunProjectsLocationsConfigurationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConfigurationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/configurations', http_method='GET', method_id='run.projects.locations.configurations.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='v1/{+parent}/configurations', request_field='', request_type_name='RunProjectsLocationsConfigurationsListRequest', response_type_name='ListConfigurationsResponse', supports_download=False)