from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudiot.v1 import cloudiot_v1_messages as messages
def ModifyCloudToDeviceConfig(self, request, global_params=None):
    """Modifies the configuration for the device, which is eventually sent from the Cloud IoT Core servers. Returns the modified configuration version and its metadata.

      Args:
        request: (CloudiotProjectsLocationsRegistriesDevicesModifyCloudToDeviceConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeviceConfig) The response message.
      """
    config = self.GetMethodConfig('ModifyCloudToDeviceConfig')
    return self._RunMethod(config, request, global_params=global_params)