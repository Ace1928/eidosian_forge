from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.beyondcorp.v1alpha import beyondcorp_v1alpha_messages as messages
class ProjectsLocationsAppGatewaysService(base_api.BaseApiService):
    """Service class for the projects_locations_appGateways resource."""
    _NAME = 'projects_locations_appGateways'

    def __init__(self, client):
        super(BeyondcorpV1alpha.ProjectsLocationsAppGatewaysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new AppGateway in a given project and location.

      Args:
        request: (BeyondcorpProjectsLocationsAppGatewaysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appGateways', http_method='POST', method_id='beyondcorp.projects.locations.appGateways.create', ordered_params=['parent'], path_params=['parent'], query_params=['appGatewayId', 'requestId', 'validateOnly'], relative_path='v1alpha/{+parent}/appGateways', request_field='appGateway', request_type_name='BeyondcorpProjectsLocationsAppGatewaysCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single AppGateway.

      Args:
        request: (BeyondcorpProjectsLocationsAppGatewaysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appGateways/{appGatewaysId}', http_method='DELETE', method_id='beyondcorp.projects.locations.appGateways.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'validateOnly'], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpProjectsLocationsAppGatewaysDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single AppGateway.

      Args:
        request: (BeyondcorpProjectsLocationsAppGatewaysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AppGateway) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appGateways/{appGatewaysId}', http_method='GET', method_id='beyondcorp.projects.locations.appGateways.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpProjectsLocationsAppGatewaysGetRequest', response_type_name='AppGateway', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (BeyondcorpProjectsLocationsAppGatewaysGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appGateways/{appGatewaysId}:getIamPolicy', http_method='GET', method_id='beyondcorp.projects.locations.appGateways.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha/{+resource}:getIamPolicy', request_field='', request_type_name='BeyondcorpProjectsLocationsAppGatewaysGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists AppGateways in a given project and location.

      Args:
        request: (BeyondcorpProjectsLocationsAppGatewaysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAppGatewaysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appGateways', http_method='GET', method_id='beyondcorp.projects.locations.appGateways.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/appGateways', request_field='', request_type_name='BeyondcorpProjectsLocationsAppGatewaysListRequest', response_type_name='ListAppGatewaysResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (BeyondcorpProjectsLocationsAppGatewaysSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appGateways/{appGatewaysId}:setIamPolicy', http_method='POST', method_id='beyondcorp.projects.locations.appGateways.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='BeyondcorpProjectsLocationsAppGatewaysSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (BeyondcorpProjectsLocationsAppGatewaysTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appGateways/{appGatewaysId}:testIamPermissions', http_method='POST', method_id='beyondcorp.projects.locations.appGateways.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='BeyondcorpProjectsLocationsAppGatewaysTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)