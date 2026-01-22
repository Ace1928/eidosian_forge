from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1 import cloudkms_v1_messages as messages
def GenerateRandomBytes(self, request, global_params=None):
    """Generate random bytes using the Cloud KMS randomness source in the provided location.

      Args:
        request: (CloudkmsProjectsLocationsGenerateRandomBytesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateRandomBytesResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateRandomBytes')
    return self._RunMethod(config, request, global_params=global_params)