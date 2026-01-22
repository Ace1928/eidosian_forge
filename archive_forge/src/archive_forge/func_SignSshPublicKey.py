from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.oslogin.v1beta import oslogin_v1beta_messages as messages
def SignSshPublicKey(self, request, global_params=None):
    """Signs an SSH public key for a user to authenticate to an instance.

      Args:
        request: (OsloginUsersProjectsZonesSignSshPublicKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SignSshPublicKeyResponse) The response message.
      """
    config = self.GetMethodConfig('SignSshPublicKey')
    return self._RunMethod(config, request, global_params=global_params)