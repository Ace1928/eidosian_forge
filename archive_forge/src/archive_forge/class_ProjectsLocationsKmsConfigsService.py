from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.netapp.v1 import netapp_v1_messages as messages
class ProjectsLocationsKmsConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_kmsConfigs resource."""
    _NAME = 'projects_locations_kmsConfigs'

    def __init__(self, client):
        super(NetappV1.ProjectsLocationsKmsConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new KMS config.

      Args:
        request: (NetappProjectsLocationsKmsConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/kmsConfigs', http_method='POST', method_id='netapp.projects.locations.kmsConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['kmsConfigId'], relative_path='v1/{+parent}/kmsConfigs', request_field='kmsConfig', request_type_name='NetappProjectsLocationsKmsConfigsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Warning! This operation will permanently delete the Kms config.

      Args:
        request: (NetappProjectsLocationsKmsConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/kmsConfigs/{kmsConfigsId}', http_method='DELETE', method_id='netapp.projects.locations.kmsConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetappProjectsLocationsKmsConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Encrypt(self, request, global_params=None):
        """Encrypt the existing volumes without CMEK encryption with the desired the KMS config for the whole region.

      Args:
        request: (NetappProjectsLocationsKmsConfigsEncryptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Encrypt')
        return self._RunMethod(config, request, global_params=global_params)
    Encrypt.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/kmsConfigs/{kmsConfigsId}:encrypt', http_method='POST', method_id='netapp.projects.locations.kmsConfigs.encrypt', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:encrypt', request_field='encryptVolumesRequest', request_type_name='NetappProjectsLocationsKmsConfigsEncryptRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the description of the specified KMS config by kms_config_id.

      Args:
        request: (NetappProjectsLocationsKmsConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (KmsConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/kmsConfigs/{kmsConfigsId}', http_method='GET', method_id='netapp.projects.locations.kmsConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetappProjectsLocationsKmsConfigsGetRequest', response_type_name='KmsConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Returns descriptions of all KMS configs owned by the caller.

      Args:
        request: (NetappProjectsLocationsKmsConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListKmsConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/kmsConfigs', http_method='GET', method_id='netapp.projects.locations.kmsConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/kmsConfigs', request_field='', request_type_name='NetappProjectsLocationsKmsConfigsListRequest', response_type_name='ListKmsConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the Kms config properties with the full spec.

      Args:
        request: (NetappProjectsLocationsKmsConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/kmsConfigs/{kmsConfigsId}', http_method='PATCH', method_id='netapp.projects.locations.kmsConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='kmsConfig', request_type_name='NetappProjectsLocationsKmsConfigsPatchRequest', response_type_name='Operation', supports_download=False)

    def Verify(self, request, global_params=None):
        """Verifies KMS config reachability.

      Args:
        request: (NetappProjectsLocationsKmsConfigsVerifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VerifyKmsConfigResponse) The response message.
      """
        config = self.GetMethodConfig('Verify')
        return self._RunMethod(config, request, global_params=global_params)
    Verify.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/kmsConfigs/{kmsConfigsId}:verify', http_method='POST', method_id='netapp.projects.locations.kmsConfigs.verify', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:verify', request_field='verifyKmsConfigRequest', request_type_name='NetappProjectsLocationsKmsConfigsVerifyRequest', response_type_name='VerifyKmsConfigResponse', supports_download=False)