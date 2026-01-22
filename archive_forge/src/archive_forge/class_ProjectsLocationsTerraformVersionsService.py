from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.config.v1alpha2 import config_v1alpha2_messages as messages
class ProjectsLocationsTerraformVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_terraformVersions resource."""
    _NAME = 'projects_locations_terraformVersions'

    def __init__(self, client):
        super(ConfigV1alpha2.ProjectsLocationsTerraformVersionsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details about a TerraformVersion.

      Args:
        request: (ConfigProjectsLocationsTerraformVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TerraformVersion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/terraformVersions/{terraformVersionsId}', http_method='GET', method_id='config.projects.locations.terraformVersions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ConfigProjectsLocationsTerraformVersionsGetRequest', response_type_name='TerraformVersion', supports_download=False)

    def List(self, request, global_params=None):
        """Lists TerraformVersions in a given project and location.

      Args:
        request: (ConfigProjectsLocationsTerraformVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTerraformVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/terraformVersions', http_method='GET', method_id='config.projects.locations.terraformVersions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/terraformVersions', request_field='', request_type_name='ConfigProjectsLocationsTerraformVersionsListRequest', response_type_name='ListTerraformVersionsResponse', supports_download=False)