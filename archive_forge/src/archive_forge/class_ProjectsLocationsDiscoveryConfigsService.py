from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class ProjectsLocationsDiscoveryConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_discoveryConfigs resource."""
    _NAME = 'projects_locations_discoveryConfigs'

    def __init__(self, client):
        super(DlpV2.ProjectsLocationsDiscoveryConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a config for discovery to scan and profile storage.

      Args:
        request: (DlpProjectsLocationsDiscoveryConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DiscoveryConfig) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/discoveryConfigs', http_method='POST', method_id='dlp.projects.locations.discoveryConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/discoveryConfigs', request_field='googlePrivacyDlpV2CreateDiscoveryConfigRequest', request_type_name='DlpProjectsLocationsDiscoveryConfigsCreateRequest', response_type_name='GooglePrivacyDlpV2DiscoveryConfig', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a discovery configuration.

      Args:
        request: (DlpProjectsLocationsDiscoveryConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/discoveryConfigs/{discoveryConfigsId}', http_method='DELETE', method_id='dlp.projects.locations.discoveryConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsLocationsDiscoveryConfigsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a discovery configuration.

      Args:
        request: (DlpProjectsLocationsDiscoveryConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DiscoveryConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/discoveryConfigs/{discoveryConfigsId}', http_method='GET', method_id='dlp.projects.locations.discoveryConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsLocationsDiscoveryConfigsGetRequest', response_type_name='GooglePrivacyDlpV2DiscoveryConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists discovery configurations.

      Args:
        request: (DlpProjectsLocationsDiscoveryConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListDiscoveryConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/discoveryConfigs', http_method='GET', method_id='dlp.projects.locations.discoveryConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/discoveryConfigs', request_field='', request_type_name='DlpProjectsLocationsDiscoveryConfigsListRequest', response_type_name='GooglePrivacyDlpV2ListDiscoveryConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a discovery configuration.

      Args:
        request: (DlpProjectsLocationsDiscoveryConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DiscoveryConfig) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/discoveryConfigs/{discoveryConfigsId}', http_method='PATCH', method_id='dlp.projects.locations.discoveryConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='googlePrivacyDlpV2UpdateDiscoveryConfigRequest', request_type_name='DlpProjectsLocationsDiscoveryConfigsPatchRequest', response_type_name='GooglePrivacyDlpV2DiscoveryConfig', supports_download=False)