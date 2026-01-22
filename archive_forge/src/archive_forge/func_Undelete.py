from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privateca.v1 import privateca_v1_messages as messages
def Undelete(self, request, global_params=None):
    """Undelete a CertificateAuthority that has been deleted.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Undelete')
    return self._RunMethod(config, request, global_params=global_params)