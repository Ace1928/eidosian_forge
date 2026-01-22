from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class LocationsWorkforcePoolsProvidersKeysOperationsService(base_api.BaseApiService):
    """Service class for the locations_workforcePools_providers_keys_operations resource."""
    _NAME = 'locations_workforcePools_providers_keys_operations'

    def __init__(self, client):
        super(IamV1.LocationsWorkforcePoolsProvidersKeysOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (IamLocationsWorkforcePoolsProvidersKeysOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/providers/{providersId}/keys/{keysId}/operations/{operationsId}', http_method='GET', method_id='iam.locations.workforcePools.providers.keys.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamLocationsWorkforcePoolsProvidersKeysOperationsGetRequest', response_type_name='Operation', supports_download=False)