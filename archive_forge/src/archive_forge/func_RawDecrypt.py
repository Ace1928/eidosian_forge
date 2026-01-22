from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1 import cloudkms_v1_messages as messages
def RawDecrypt(self, request, global_params=None):
    """Decrypts data that was originally encrypted using a raw cryptographic mechanism. The CryptoKey.purpose must be RAW_ENCRYPT_DECRYPT.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRawDecryptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RawDecryptResponse) The response message.
      """
    config = self.GetMethodConfig('RawDecrypt')
    return self._RunMethod(config, request, global_params=global_params)