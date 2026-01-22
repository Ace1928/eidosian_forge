from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
def GetOwnerInstance(self, request, global_params=None):
    """Find owner instance from given ip address.

      Args:
        request: (ComputeGlobalAddressesGetOwnerInstanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetOwnerInstanceResponse) The response message.
      """
    config = self.GetMethodConfig('GetOwnerInstance')
    return self._RunMethod(config, request, global_params=global_params)