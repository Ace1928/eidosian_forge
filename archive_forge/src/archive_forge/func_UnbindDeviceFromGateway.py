from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudiot.v1 import cloudiot_v1_messages as messages
def UnbindDeviceFromGateway(self, request, global_params=None):
    """Deletes the association between the device and the gateway.

      Args:
        request: (CloudiotProjectsLocationsRegistriesUnbindDeviceFromGatewayRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UnbindDeviceFromGatewayResponse) The response message.
      """
    config = self.GetMethodConfig('UnbindDeviceFromGateway')
    return self._RunMethod(config, request, global_params=global_params)