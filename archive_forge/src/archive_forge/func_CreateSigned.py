from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
def CreateSigned(self, request, global_params=None):
    """Creates a signed device under a node or customer.

      Args:
        request: (SasportalNodesNodesDevicesCreateSignedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
    config = self.GetMethodConfig('CreateSigned')
    return self._RunMethod(config, request, global_params=global_params)