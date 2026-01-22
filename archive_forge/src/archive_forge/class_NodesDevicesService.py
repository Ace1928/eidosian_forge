from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
class NodesDevicesService(base_api.BaseApiService):
    """Service class for the nodes_devices resource."""
    _NAME = 'nodes_devices'

    def __init__(self, client):
        super(SasportalV1alpha1.NodesDevicesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a device under a node or customer.

      Args:
        request: (SasportalNodesDevicesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/devices', http_method='POST', method_id='sasportal.nodes.devices.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/devices', request_field='sasPortalDevice', request_type_name='SasportalNodesDevicesCreateRequest', response_type_name='SasPortalDevice', supports_download=False)

    def CreateSigned(self, request, global_params=None):
        """Creates a signed device under a node or customer.

      Args:
        request: (SasportalNodesDevicesCreateSignedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('CreateSigned')
        return self._RunMethod(config, request, global_params=global_params)
    CreateSigned.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/devices:createSigned', http_method='POST', method_id='sasportal.nodes.devices.createSigned', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/devices:createSigned', request_field='sasPortalCreateSignedDeviceRequest', request_type_name='SasportalNodesDevicesCreateSignedRequest', response_type_name='SasPortalDevice', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a device.

      Args:
        request: (SasportalNodesDevicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/devices/{devicesId}', http_method='DELETE', method_id='sasportal.nodes.devices.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SasportalNodesDevicesDeleteRequest', response_type_name='SasPortalEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details about a device.

      Args:
        request: (SasportalNodesDevicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/devices/{devicesId}', http_method='GET', method_id='sasportal.nodes.devices.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SasportalNodesDevicesGetRequest', response_type_name='SasPortalDevice', supports_download=False)

    def List(self, request, global_params=None):
        """Lists devices under a node or customer.

      Args:
        request: (SasportalNodesDevicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalListDevicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/devices', http_method='GET', method_id='sasportal.nodes.devices.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/devices', request_field='', request_type_name='SasportalNodesDevicesListRequest', response_type_name='SasPortalListDevicesResponse', supports_download=False)

    def Move(self, request, global_params=None):
        """Moves a device under another node or customer.

      Args:
        request: (SasportalNodesDevicesMoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalOperation) The response message.
      """
        config = self.GetMethodConfig('Move')
        return self._RunMethod(config, request, global_params=global_params)
    Move.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/devices/{devicesId}:move', http_method='POST', method_id='sasportal.nodes.devices.move', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:move', request_field='sasPortalMoveDeviceRequest', request_type_name='SasportalNodesDevicesMoveRequest', response_type_name='SasPortalOperation', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a device.

      Args:
        request: (SasportalNodesDevicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/devices/{devicesId}', http_method='PATCH', method_id='sasportal.nodes.devices.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='sasPortalDevice', request_type_name='SasportalNodesDevicesPatchRequest', response_type_name='SasPortalDevice', supports_download=False)

    def SignDevice(self, request, global_params=None):
        """Signs a device.

      Args:
        request: (SasportalNodesDevicesSignDeviceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalEmpty) The response message.
      """
        config = self.GetMethodConfig('SignDevice')
        return self._RunMethod(config, request, global_params=global_params)
    SignDevice.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/devices/{devicesId}:signDevice', http_method='POST', method_id='sasportal.nodes.devices.signDevice', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:signDevice', request_field='sasPortalSignDeviceRequest', request_type_name='SasportalNodesDevicesSignDeviceRequest', response_type_name='SasPortalEmpty', supports_download=False)

    def UpdateSigned(self, request, global_params=None):
        """Updates a signed device.

      Args:
        request: (SasportalNodesDevicesUpdateSignedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('UpdateSigned')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateSigned.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/devices/{devicesId}:updateSigned', http_method='PATCH', method_id='sasportal.nodes.devices.updateSigned', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:updateSigned', request_field='sasPortalUpdateSignedDeviceRequest', request_type_name='SasportalNodesDevicesUpdateSignedRequest', response_type_name='SasPortalDevice', supports_download=False)