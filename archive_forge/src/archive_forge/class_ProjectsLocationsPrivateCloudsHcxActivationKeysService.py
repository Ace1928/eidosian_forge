from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsPrivateCloudsHcxActivationKeysService(base_api.BaseApiService):
    """Service class for the projects_locations_privateClouds_hcxActivationKeys resource."""
    _NAME = 'projects_locations_privateClouds_hcxActivationKeys'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsPrivateCloudsHcxActivationKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new HCX activation key in a given private cloud.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/hcxActivationKeys', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.hcxActivationKeys.create', ordered_params=['parent'], path_params=['parent'], query_params=['hcxActivationKeyId', 'requestId'], relative_path='v1/{+parent}/hcxActivationKeys', request_field='hcxActivationKey', request_type_name='VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysCreateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a `HcxActivationKey` resource by its resource name.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HcxActivationKey) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/hcxActivationKeys/{hcxActivationKeysId}', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.hcxActivationKeys.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysGetRequest', response_type_name='HcxActivationKey', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/hcxActivationKeys/{hcxActivationKeysId}:getIamPolicy', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.hcxActivationKeys.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `HcxActivationKey` resources in a given private cloud.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListHcxActivationKeysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/hcxActivationKeys', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.hcxActivationKeys.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/hcxActivationKeys', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysListRequest', response_type_name='ListHcxActivationKeysResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/hcxActivationKeys/{hcxActivationKeysId}:setIamPolicy', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.hcxActivationKeys.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/hcxActivationKeys/{hcxActivationKeysId}:testIamPermissions', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.hcxActivationKeys.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='VmwareengineProjectsLocationsPrivateCloudsHcxActivationKeysTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)