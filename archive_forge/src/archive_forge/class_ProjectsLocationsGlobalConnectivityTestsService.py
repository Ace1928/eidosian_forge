from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkmanagement.v1 import networkmanagement_v1_messages as messages
class ProjectsLocationsGlobalConnectivityTestsService(base_api.BaseApiService):
    """Service class for the projects_locations_global_connectivityTests resource."""
    _NAME = 'projects_locations_global_connectivityTests'

    def __init__(self, client):
        super(NetworkmanagementV1.ProjectsLocationsGlobalConnectivityTestsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Connectivity Test. After you create a test, the reachability analysis is performed as part of the long running operation, which completes when the analysis completes. If the endpoint specifications in `ConnectivityTest` are invalid (for example, containing non-existent resources in the network, or you don't have read permissions to the network configurations of listed projects), then the reachability result returns a value of `UNKNOWN`. If the endpoint specifications in `ConnectivityTest` are incomplete, the reachability result returns a value of AMBIGUOUS. For more information, see the Connectivity Test documentation.

      Args:
        request: (NetworkmanagementProjectsLocationsGlobalConnectivityTestsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/connectivityTests', http_method='POST', method_id='networkmanagement.projects.locations.global.connectivityTests.create', ordered_params=['parent'], path_params=['parent'], query_params=['testId'], relative_path='v1/{+parent}/connectivityTests', request_field='connectivityTest', request_type_name='NetworkmanagementProjectsLocationsGlobalConnectivityTestsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a specific `ConnectivityTest`.

      Args:
        request: (NetworkmanagementProjectsLocationsGlobalConnectivityTestsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/connectivityTests/{connectivityTestsId}', http_method='DELETE', method_id='networkmanagement.projects.locations.global.connectivityTests.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkmanagementProjectsLocationsGlobalConnectivityTestsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a specific Connectivity Test.

      Args:
        request: (NetworkmanagementProjectsLocationsGlobalConnectivityTestsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConnectivityTest) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/connectivityTests/{connectivityTestsId}', http_method='GET', method_id='networkmanagement.projects.locations.global.connectivityTests.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkmanagementProjectsLocationsGlobalConnectivityTestsGetRequest', response_type_name='ConnectivityTest', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworkmanagementProjectsLocationsGlobalConnectivityTestsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/connectivityTests/{connectivityTestsId}:getIamPolicy', http_method='GET', method_id='networkmanagement.projects.locations.global.connectivityTests.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='NetworkmanagementProjectsLocationsGlobalConnectivityTestsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all Connectivity Tests owned by a project.

      Args:
        request: (NetworkmanagementProjectsLocationsGlobalConnectivityTestsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConnectivityTestsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/connectivityTests', http_method='GET', method_id='networkmanagement.projects.locations.global.connectivityTests.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/connectivityTests', request_field='', request_type_name='NetworkmanagementProjectsLocationsGlobalConnectivityTestsListRequest', response_type_name='ListConnectivityTestsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the configuration of an existing `ConnectivityTest`. After you update a test, the reachability analysis is performed as part of the long running operation, which completes when the analysis completes. The Reachability state in the test resource is updated with the new result. If the endpoint specifications in `ConnectivityTest` are invalid (for example, they contain non-existent resources in the network, or the user does not have read permissions to the network configurations of listed projects), then the reachability result returns a value of UNKNOWN. If the endpoint specifications in `ConnectivityTest` are incomplete, the reachability result returns a value of `AMBIGUOUS`. See the documentation in `ConnectivityTest` for for more details.

      Args:
        request: (NetworkmanagementProjectsLocationsGlobalConnectivityTestsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/connectivityTests/{connectivityTestsId}', http_method='PATCH', method_id='networkmanagement.projects.locations.global.connectivityTests.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='connectivityTest', request_type_name='NetworkmanagementProjectsLocationsGlobalConnectivityTestsPatchRequest', response_type_name='Operation', supports_download=False)

    def Rerun(self, request, global_params=None):
        """Rerun an existing `ConnectivityTest`. After the user triggers the rerun, the reachability analysis is performed as part of the long running operation, which completes when the analysis completes. Even though the test configuration remains the same, the reachability result may change due to underlying network configuration changes. If the endpoint specifications in `ConnectivityTest` become invalid (for example, specified resources are deleted in the network, or you lost read permissions to the network configurations of listed projects), then the reachability result returns a value of `UNKNOWN`.

      Args:
        request: (NetworkmanagementProjectsLocationsGlobalConnectivityTestsRerunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Rerun')
        return self._RunMethod(config, request, global_params=global_params)
    Rerun.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/connectivityTests/{connectivityTestsId}:rerun', http_method='POST', method_id='networkmanagement.projects.locations.global.connectivityTests.rerun', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:rerun', request_field='rerunConnectivityTestRequest', request_type_name='NetworkmanagementProjectsLocationsGlobalConnectivityTestsRerunRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworkmanagementProjectsLocationsGlobalConnectivityTestsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/connectivityTests/{connectivityTestsId}:setIamPolicy', http_method='POST', method_id='networkmanagement.projects.locations.global.connectivityTests.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='NetworkmanagementProjectsLocationsGlobalConnectivityTestsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworkmanagementProjectsLocationsGlobalConnectivityTestsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/connectivityTests/{connectivityTestsId}:testIamPermissions', http_method='POST', method_id='networkmanagement.projects.locations.global.connectivityTests.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='NetworkmanagementProjectsLocationsGlobalConnectivityTestsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)