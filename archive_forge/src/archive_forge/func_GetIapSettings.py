from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iap.v1 import iap_v1_messages as messages
def GetIapSettings(self, request, global_params=None):
    """Gets the IAP settings on a particular IAP protected resource.

      Args:
        request: (IapGetIapSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IapSettings) The response message.
      """
    config = self.GetMethodConfig('GetIapSettings')
    return self._RunMethod(config, request, global_params=global_params)