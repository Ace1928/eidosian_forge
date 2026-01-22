from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
def MoveDevicesToOu(self, request, global_params=None):
    """Move or insert multiple Chrome OS Devices to organizational unit.

      Args:
        request: (DirectoryChromeosdevicesMoveDevicesToOuRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryChromeosdevicesMoveDevicesToOuResponse) The response message.
      """
    config = self.GetMethodConfig('MoveDevicesToOu')
    return self._RunMethod(config, request, global_params=global_params)