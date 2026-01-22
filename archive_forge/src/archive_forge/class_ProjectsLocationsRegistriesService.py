from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudiot.v1 import cloudiot_v1_messages as messages
class ProjectsLocationsRegistriesService(base_api.BaseApiService):
    """Service class for the projects_locations_registries resource."""
    _NAME = 'projects_locations_registries'

    def __init__(self, client):
        super(CloudiotV1.ProjectsLocationsRegistriesService, self).__init__(client)
        self._upload_configs = {}

    def BindDeviceToGateway(self, request, global_params=None):
        """Associates the device with the gateway.

      Args:
        request: (CloudiotProjectsLocationsRegistriesBindDeviceToGatewayRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BindDeviceToGatewayResponse) The response message.
      """
        config = self.GetMethodConfig('BindDeviceToGateway')
        return self._RunMethod(config, request, global_params=global_params)
    BindDeviceToGateway.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}:bindDeviceToGateway', http_method='POST', method_id='cloudiot.projects.locations.registries.bindDeviceToGateway', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:bindDeviceToGateway', request_field='bindDeviceToGatewayRequest', request_type_name='CloudiotProjectsLocationsRegistriesBindDeviceToGatewayRequest', response_type_name='BindDeviceToGatewayResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a device registry that contains devices.

      Args:
        request: (CloudiotProjectsLocationsRegistriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeviceRegistry) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries', http_method='POST', method_id='cloudiot.projects.locations.registries.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/registries', request_field='deviceRegistry', request_type_name='CloudiotProjectsLocationsRegistriesCreateRequest', response_type_name='DeviceRegistry', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a device registry configuration.

      Args:
        request: (CloudiotProjectsLocationsRegistriesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}', http_method='DELETE', method_id='cloudiot.projects.locations.registries.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudiotProjectsLocationsRegistriesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a device registry configuration.

      Args:
        request: (CloudiotProjectsLocationsRegistriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeviceRegistry) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}', http_method='GET', method_id='cloudiot.projects.locations.registries.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudiotProjectsLocationsRegistriesGetRequest', response_type_name='DeviceRegistry', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (CloudiotProjectsLocationsRegistriesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}:getIamPolicy', http_method='POST', method_id='cloudiot.projects.locations.registries.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='CloudiotProjectsLocationsRegistriesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists device registries.

      Args:
        request: (CloudiotProjectsLocationsRegistriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDeviceRegistriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries', http_method='GET', method_id='cloudiot.projects.locations.registries.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/registries', request_field='', request_type_name='CloudiotProjectsLocationsRegistriesListRequest', response_type_name='ListDeviceRegistriesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a device registry configuration.

      Args:
        request: (CloudiotProjectsLocationsRegistriesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeviceRegistry) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}', http_method='PATCH', method_id='cloudiot.projects.locations.registries.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='deviceRegistry', request_type_name='CloudiotProjectsLocationsRegistriesPatchRequest', response_type_name='DeviceRegistry', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (CloudiotProjectsLocationsRegistriesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}:setIamPolicy', http_method='POST', method_id='cloudiot.projects.locations.registries.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='CloudiotProjectsLocationsRegistriesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error.

      Args:
        request: (CloudiotProjectsLocationsRegistriesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}:testIamPermissions', http_method='POST', method_id='cloudiot.projects.locations.registries.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='CloudiotProjectsLocationsRegistriesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

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
    UnbindDeviceFromGateway.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}:unbindDeviceFromGateway', http_method='POST', method_id='cloudiot.projects.locations.registries.unbindDeviceFromGateway', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:unbindDeviceFromGateway', request_field='unbindDeviceFromGatewayRequest', request_type_name='CloudiotProjectsLocationsRegistriesUnbindDeviceFromGatewayRequest', response_type_name='UnbindDeviceFromGatewayResponse', supports_download=False)