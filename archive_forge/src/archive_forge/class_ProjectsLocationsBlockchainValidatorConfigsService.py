from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.blockchainvalidatormanager.v1alpha import blockchainvalidatormanager_v1alpha_messages as messages
class ProjectsLocationsBlockchainValidatorConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_blockchainValidatorConfigs resource."""
    _NAME = 'projects_locations_blockchainValidatorConfigs'

    def __init__(self, client):
        super(BlockchainvalidatormanagerV1alpha.ProjectsLocationsBlockchainValidatorConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new blockchain validator configuration in a given project and location.

      Args:
        request: (BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/blockchainValidatorConfigs', http_method='POST', method_id='blockchainvalidatormanager.projects.locations.blockchainValidatorConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['blockchainValidatorConfigId', 'requestId'], relative_path='v1alpha/{+parent}/blockchainValidatorConfigs', request_field='blockchainValidatorConfig', request_type_name='BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single BlockchainValidatorConfig.

      Args:
        request: (BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/blockchainValidatorConfigs/{blockchainValidatorConfigsId}', http_method='DELETE', method_id='blockchainvalidatormanager.projects.locations.blockchainValidatorConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Generate(self, request, global_params=None):
        """Create one or more blockchain validator configurations, derived based on the specification provided.

      Args:
        request: (BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsGenerateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Generate')
        return self._RunMethod(config, request, global_params=global_params)
    Generate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/blockchainValidatorConfigs:generate', http_method='POST', method_id='blockchainvalidatormanager.projects.locations.blockchainValidatorConfigs.generate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha/{+parent}/blockchainValidatorConfigs:generate', request_field='generateBlockchainValidatorConfigsRequest', request_type_name='BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsGenerateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single blockchain validator configuration.

      Args:
        request: (BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BlockchainValidatorConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/blockchainValidatorConfigs/{blockchainValidatorConfigsId}', http_method='GET', method_id='blockchainvalidatormanager.projects.locations.blockchainValidatorConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsGetRequest', response_type_name='BlockchainValidatorConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists BlockchainValidatorConfigs in a given project and location.

      Args:
        request: (BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBlockchainValidatorConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/blockchainValidatorConfigs', http_method='GET', method_id='blockchainvalidatormanager.projects.locations.blockchainValidatorConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/blockchainValidatorConfigs', request_field='', request_type_name='BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsListRequest', response_type_name='ListBlockchainValidatorConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single blockchain validator configuration.

      Args:
        request: (BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/blockchainValidatorConfigs/{blockchainValidatorConfigsId}', http_method='PATCH', method_id='blockchainvalidatormanager.projects.locations.blockchainValidatorConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='blockchainValidatorConfig', request_type_name='BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsPatchRequest', response_type_name='Operation', supports_download=False)