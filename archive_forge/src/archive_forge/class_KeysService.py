from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apikeys.v2 import apikeys_v2_messages as messages
class KeysService(base_api.BaseApiService):
    """Service class for the keys resource."""
    _NAME = 'keys'

    def __init__(self, client):
        super(ApikeysV2.KeysService, self).__init__(client)
        self._upload_configs = {}

    def LookupKey(self, request, global_params=None):
        """Find the parent project and resource name of the API key that matches the key string in the request. If the API key has been purged, resource name will not be set. The service account must have the `apikeys.keys.lookup` permission on the parent project.

      Args:
        request: (ApikeysKeysLookupKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (V2LookupKeyResponse) The response message.
      """
        config = self.GetMethodConfig('LookupKey')
        return self._RunMethod(config, request, global_params=global_params)
    LookupKey.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='apikeys.keys.lookupKey', ordered_params=[], path_params=[], query_params=['keyString'], relative_path='v2/keys:lookupKey', request_field='', request_type_name='ApikeysKeysLookupKeyRequest', response_type_name='V2LookupKeyResponse', supports_download=False)