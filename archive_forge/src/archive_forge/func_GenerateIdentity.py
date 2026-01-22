from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v1alpha import serviceusage_v1alpha_messages as messages
def GenerateIdentity(self, request, global_params=None):
    """Generate service identity for service.

      Args:
        request: (ServiceusageServicesGenerateIdentityRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('GenerateIdentity')
    return self._RunMethod(config, request, global_params=global_params)