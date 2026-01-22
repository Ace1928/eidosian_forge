from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudiot.v1 import cloudiot_v1_messages as messages
def SendCommandToDevice(self, request, global_params=None):
    """Sends a command to the specified device. In order for a device to be able to receive commands, it must: 1) be connected to Cloud IoT Core using the MQTT protocol, and 2) be subscribed to the group of MQTT topics specified by /devices/{device-id}/commands/#. This subscription will receive commands at the top-level topic /devices/{device-id}/commands as well as commands for subfolders, like /devices/{device-id}/commands/subfolder. Note that subscribing to specific subfolders is not supported. If the command could not be delivered to the device, this method will return an error; in particular, if the device is not subscribed, this method will return FAILED_PRECONDITION. Otherwise, this method will return OK. If the subscription is QoS 1, at least once delivery will be guaranteed; for QoS 0, no acknowledgment will be expected from the device.

      Args:
        request: (CloudiotProjectsLocationsRegistriesDevicesSendCommandToDeviceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SendCommandToDeviceResponse) The response message.
      """
    config = self.GetMethodConfig('SendCommandToDevice')
    return self._RunMethod(config, request, global_params=global_params)