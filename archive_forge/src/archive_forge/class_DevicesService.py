from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
class DevicesService(base_api.BaseApiService):
    """Service class for the devices resource."""
    _NAME = 'devices'

    def __init__(self, client):
        super(CloudidentityV1.DevicesService, self).__init__(client)
        self._upload_configs = {}

    def CancelWipe(self, request, global_params=None):
        """Cancels an unfinished device wipe. This operation can be used to cancel device wipe in the gap between the wipe operation returning success and the device being wiped. This operation is possible when the device is in a "pending wipe" state. The device enters the "pending wipe" state when a wipe device command is issued, but has not yet been sent to the device. The cancel wipe will fail if the wipe command has already been issued to the device.

      Args:
        request: (CloudidentityDevicesCancelWipeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CancelWipe')
        return self._RunMethod(config, request, global_params=global_params)
    CancelWipe.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}:cancelWipe', http_method='POST', method_id='cloudidentity.devices.cancelWipe', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancelWipe', request_field='googleAppsCloudidentityDevicesV1CancelWipeDeviceRequest', request_type_name='CloudidentityDevicesCancelWipeRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a device. Only company-owned device may be created. **Note**: This method is available only to customers who have one of the following SKUs: Enterprise Standard, Enterprise Plus, Enterprise for Education, and Cloud Identity Premium.

      Args:
        request: (CloudidentityDevicesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudidentity.devices.create', ordered_params=[], path_params=[], query_params=['customer'], relative_path='v1/devices', request_field='googleAppsCloudidentityDevicesV1Device', request_type_name='CloudidentityDevicesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified device.

      Args:
        request: (CloudidentityDevicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}', http_method='DELETE', method_id='cloudidentity.devices.delete', ordered_params=['name'], path_params=['name'], query_params=['customer'], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityDevicesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified device.

      Args:
        request: (CloudidentityDevicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleAppsCloudidentityDevicesV1Device) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}', http_method='GET', method_id='cloudidentity.devices.get', ordered_params=['name'], path_params=['name'], query_params=['customer'], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityDevicesGetRequest', response_type_name='GoogleAppsCloudidentityDevicesV1Device', supports_download=False)

    def List(self, request, global_params=None):
        """Lists/Searches devices.

      Args:
        request: (CloudidentityDevicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleAppsCloudidentityDevicesV1ListDevicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudidentity.devices.list', ordered_params=[], path_params=[], query_params=['customer', 'filter', 'orderBy', 'pageSize', 'pageToken', 'view'], relative_path='v1/devices', request_field='', request_type_name='CloudidentityDevicesListRequest', response_type_name='GoogleAppsCloudidentityDevicesV1ListDevicesResponse', supports_download=False)

    def Wipe(self, request, global_params=None):
        """Wipes all data on the specified device.

      Args:
        request: (CloudidentityDevicesWipeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Wipe')
        return self._RunMethod(config, request, global_params=global_params)
    Wipe.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/devices/{devicesId}:wipe', http_method='POST', method_id='cloudidentity.devices.wipe', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:wipe', request_field='googleAppsCloudidentityDevicesV1WipeDeviceRequest', request_type_name='CloudidentityDevicesWipeRequest', response_type_name='Operation', supports_download=False)