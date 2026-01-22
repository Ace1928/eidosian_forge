from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.backupdr.v1 import backupdr_v1_messages as messages
class ProjectsLocationsManagementServersService(base_api.BaseApiService):
    """Service class for the projects_locations_managementServers resource."""
    _NAME = 'projects_locations_managementServers'

    def __init__(self, client):
        super(BackupdrV1.ProjectsLocationsManagementServersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ManagementServer in a given project and location.

      Args:
        request: (BackupdrProjectsLocationsManagementServersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/managementServers', http_method='POST', method_id='backupdr.projects.locations.managementServers.create', ordered_params=['parent'], path_params=['parent'], query_params=['managementServerId', 'requestId'], relative_path='v1/{+parent}/managementServers', request_field='managementServer', request_type_name='BackupdrProjectsLocationsManagementServersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ManagementServer.

      Args:
        request: (BackupdrProjectsLocationsManagementServersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/managementServers/{managementServersId}', http_method='DELETE', method_id='backupdr.projects.locations.managementServers.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='BackupdrProjectsLocationsManagementServersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ManagementServer.

      Args:
        request: (BackupdrProjectsLocationsManagementServersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagementServer) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/managementServers/{managementServersId}', http_method='GET', method_id='backupdr.projects.locations.managementServers.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BackupdrProjectsLocationsManagementServersGetRequest', response_type_name='ManagementServer', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (BackupdrProjectsLocationsManagementServersGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/managementServers/{managementServersId}:getIamPolicy', http_method='GET', method_id='backupdr.projects.locations.managementServers.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='BackupdrProjectsLocationsManagementServersGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ManagementServers in a given project and location.

      Args:
        request: (BackupdrProjectsLocationsManagementServersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListManagementServersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/managementServers', http_method='GET', method_id='backupdr.projects.locations.managementServers.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/managementServers', request_field='', request_type_name='BackupdrProjectsLocationsManagementServersListRequest', response_type_name='ListManagementServersResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (BackupdrProjectsLocationsManagementServersSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/managementServers/{managementServersId}:setIamPolicy', http_method='POST', method_id='backupdr.projects.locations.managementServers.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='BackupdrProjectsLocationsManagementServersSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (BackupdrProjectsLocationsManagementServersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/managementServers/{managementServersId}:testIamPermissions', http_method='POST', method_id='backupdr.projects.locations.managementServers.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='BackupdrProjectsLocationsManagementServersTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)