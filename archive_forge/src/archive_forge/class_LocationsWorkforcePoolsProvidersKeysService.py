from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class LocationsWorkforcePoolsProvidersKeysService(base_api.BaseApiService):
    """Service class for the locations_workforcePools_providers_keys resource."""
    _NAME = 'locations_workforcePools_providers_keys'

    def __init__(self, client):
        super(IamV1.LocationsWorkforcePoolsProvidersKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new WorkforcePoolProviderKey in a WorkforcePoolProvider.

      Args:
        request: (IamLocationsWorkforcePoolsProvidersKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/providers/{providersId}/keys', http_method='POST', method_id='iam.locations.workforcePools.providers.keys.create', ordered_params=['parent'], path_params=['parent'], query_params=['workforcePoolProviderKeyId'], relative_path='v1/{+parent}/keys', request_field='workforcePoolProviderKey', request_type_name='IamLocationsWorkforcePoolsProvidersKeysCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a WorkforcePoolProviderKey. You can undelete a key for 30 days. After 30 days, deletion is permanent.

      Args:
        request: (IamLocationsWorkforcePoolsProvidersKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/providers/{providersId}/keys/{keysId}', http_method='DELETE', method_id='iam.locations.workforcePools.providers.keys.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamLocationsWorkforcePoolsProvidersKeysDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a WorkforcePoolProviderKey.

      Args:
        request: (IamLocationsWorkforcePoolsProvidersKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkforcePoolProviderKey) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/providers/{providersId}/keys/{keysId}', http_method='GET', method_id='iam.locations.workforcePools.providers.keys.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamLocationsWorkforcePoolsProvidersKeysGetRequest', response_type_name='WorkforcePoolProviderKey', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all non-deleted WorkforcePoolProviderKeys in a WorkforcePoolProvider. If `show_deleted` is set to `true`, then deleted keys are also listed.

      Args:
        request: (IamLocationsWorkforcePoolsProvidersKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkforcePoolProviderKeysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/providers/{providersId}/keys', http_method='GET', method_id='iam.locations.workforcePools.providers.keys.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v1/{+parent}/keys', request_field='', request_type_name='IamLocationsWorkforcePoolsProvidersKeysListRequest', response_type_name='ListWorkforcePoolProviderKeysResponse', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a WorkforcePoolProviderKey, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (IamLocationsWorkforcePoolsProvidersKeysUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/providers/{providersId}/keys/{keysId}:undelete', http_method='POST', method_id='iam.locations.workforcePools.providers.keys.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeleteWorkforcePoolProviderKeyRequest', request_type_name='IamLocationsWorkforcePoolsProvidersKeysUndeleteRequest', response_type_name='Operation', supports_download=False)