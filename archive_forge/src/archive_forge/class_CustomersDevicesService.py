from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
class CustomersDevicesService(base_api.BaseApiService):
    """Service class for the customers_devices resource."""
    _NAME = 'customers_devices'

    def __init__(self, client):
        super(SasportalV1alpha1.CustomersDevicesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a device under a node or customer.

      Args:
        request: (SasportalCustomersDevicesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/devices', http_method='POST', method_id='sasportal.customers.devices.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/devices', request_field='sasPortalDevice', request_type_name='SasportalCustomersDevicesCreateRequest', response_type_name='SasPortalDevice', supports_download=False)

    def CreateSigned(self, request, global_params=None):
        """Creates a signed device under a node or customer.

      Args:
        request: (SasportalCustomersDevicesCreateSignedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('CreateSigned')
        return self._RunMethod(config, request, global_params=global_params)
    CreateSigned.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/devices:createSigned', http_method='POST', method_id='sasportal.customers.devices.createSigned', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/devices:createSigned', request_field='sasPortalCreateSignedDeviceRequest', request_type_name='SasportalCustomersDevicesCreateSignedRequest', response_type_name='SasPortalDevice', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a device.

      Args:
        request: (SasportalCustomersDevicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/devices/{devicesId}', http_method='DELETE', method_id='sasportal.customers.devices.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SasportalCustomersDevicesDeleteRequest', response_type_name='SasPortalEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details about a device.

      Args:
        request: (SasportalCustomersDevicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/devices/{devicesId}', http_method='GET', method_id='sasportal.customers.devices.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SasportalCustomersDevicesGetRequest', response_type_name='SasPortalDevice', supports_download=False)

    def List(self, request, global_params=None):
        """Lists devices under a node or customer.

      Args:
        request: (SasportalCustomersDevicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalListDevicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/devices', http_method='GET', method_id='sasportal.customers.devices.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/devices', request_field='', request_type_name='SasportalCustomersDevicesListRequest', response_type_name='SasPortalListDevicesResponse', supports_download=False)

    def Move(self, request, global_params=None):
        """Moves a device under another node or customer.

      Args:
        request: (SasportalCustomersDevicesMoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalOperation) The response message.
      """
        config = self.GetMethodConfig('Move')
        return self._RunMethod(config, request, global_params=global_params)
    Move.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/devices/{devicesId}:move', http_method='POST', method_id='sasportal.customers.devices.move', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:move', request_field='sasPortalMoveDeviceRequest', request_type_name='SasportalCustomersDevicesMoveRequest', response_type_name='SasPortalOperation', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a device.

      Args:
        request: (SasportalCustomersDevicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/devices/{devicesId}', http_method='PATCH', method_id='sasportal.customers.devices.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='sasPortalDevice', request_type_name='SasportalCustomersDevicesPatchRequest', response_type_name='SasPortalDevice', supports_download=False)

    def SignDevice(self, request, global_params=None):
        """Signs a device.

      Args:
        request: (SasportalCustomersDevicesSignDeviceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalEmpty) The response message.
      """
        config = self.GetMethodConfig('SignDevice')
        return self._RunMethod(config, request, global_params=global_params)
    SignDevice.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/devices/{devicesId}:signDevice', http_method='POST', method_id='sasportal.customers.devices.signDevice', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:signDevice', request_field='sasPortalSignDeviceRequest', request_type_name='SasportalCustomersDevicesSignDeviceRequest', response_type_name='SasPortalEmpty', supports_download=False)

    def UpdateSigned(self, request, global_params=None):
        """Updates a signed device.

      Args:
        request: (SasportalCustomersDevicesUpdateSignedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('UpdateSigned')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateSigned.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/devices/{devicesId}:updateSigned', http_method='PATCH', method_id='sasportal.customers.devices.updateSigned', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:updateSigned', request_field='sasPortalUpdateSignedDeviceRequest', request_type_name='SasportalCustomersDevicesUpdateSignedRequest', response_type_name='SasPortalDevice', supports_download=False)