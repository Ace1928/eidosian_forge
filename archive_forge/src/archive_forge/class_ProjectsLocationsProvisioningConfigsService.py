from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
class ProjectsLocationsProvisioningConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_provisioningConfigs resource."""
    _NAME = 'projects_locations_provisioningConfigs'

    def __init__(self, client):
        super(BaremetalsolutionV2.ProjectsLocationsProvisioningConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create new ProvisioningConfig.

      Args:
        request: (BaremetalsolutionProjectsLocationsProvisioningConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProvisioningConfig) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/provisioningConfigs', http_method='POST', method_id='baremetalsolution.projects.locations.provisioningConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['email'], relative_path='v2/{+parent}/provisioningConfigs', request_field='provisioningConfig', request_type_name='BaremetalsolutionProjectsLocationsProvisioningConfigsCreateRequest', response_type_name='ProvisioningConfig', supports_download=False)

    def Get(self, request, global_params=None):
        """Get ProvisioningConfig by name.

      Args:
        request: (BaremetalsolutionProjectsLocationsProvisioningConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProvisioningConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/provisioningConfigs/{provisioningConfigsId}', http_method='GET', method_id='baremetalsolution.projects.locations.provisioningConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsProvisioningConfigsGetRequest', response_type_name='ProvisioningConfig', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update existing ProvisioningConfig.

      Args:
        request: (BaremetalsolutionProjectsLocationsProvisioningConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProvisioningConfig) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/provisioningConfigs/{provisioningConfigsId}', http_method='PATCH', method_id='baremetalsolution.projects.locations.provisioningConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['email', 'updateMask'], relative_path='v2/{+name}', request_field='provisioningConfig', request_type_name='BaremetalsolutionProjectsLocationsProvisioningConfigsPatchRequest', response_type_name='ProvisioningConfig', supports_download=False)

    def Submit(self, request, global_params=None):
        """Submit a provisiong configuration for a given project.

      Args:
        request: (BaremetalsolutionProjectsLocationsProvisioningConfigsSubmitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SubmitProvisioningConfigResponse) The response message.
      """
        config = self.GetMethodConfig('Submit')
        return self._RunMethod(config, request, global_params=global_params)
    Submit.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/provisioningConfigs:submit', http_method='POST', method_id='baremetalsolution.projects.locations.provisioningConfigs.submit', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/provisioningConfigs:submit', request_field='submitProvisioningConfigRequest', request_type_name='BaremetalsolutionProjectsLocationsProvisioningConfigsSubmitRequest', response_type_name='SubmitProvisioningConfigResponse', supports_download=False)