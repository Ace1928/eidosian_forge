from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def GenerateEphemeralCert(self, request, global_params=None):
    """Generates a short-lived X509 certificate containing the provided public key and signed by a private key specific to the target instance. Users may use the certificate to authenticate as themselves when connecting to the database.

      Args:
        request: (SqlConnectGenerateEphemeralRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateEphemeralCertResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateEphemeralCert')
    return self._RunMethod(config, request, global_params=global_params)