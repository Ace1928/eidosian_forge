from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iap.v1 import iap_v1_messages as messages
def ResetSecret(self, request, global_params=None):
    """Resets an Identity Aware Proxy (IAP) OAuth client secret. Useful if the secret was compromised. Requires that the client is owned by IAP.

      Args:
        request: (IapProjectsBrandsIdentityAwareProxyClientsResetSecretRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IdentityAwareProxyClient) The response message.
      """
    config = self.GetMethodConfig('ResetSecret')
    return self._RunMethod(config, request, global_params=global_params)