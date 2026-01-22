from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workstations.v1beta import workstations_v1beta_messages as messages
class ProjectsLocationsWorkstationClustersWorkstationConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_workstationClusters_workstationConfigs resource."""
    _NAME = 'projects_locations_workstationClusters_workstationConfigs'

    def __init__(self, client):
        super(WorkstationsV1beta.ProjectsLocationsWorkstationClustersWorkstationConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new workstation configuration.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs', http_method='POST', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly', 'workstationConfigId'], relative_path='v1beta/{+parent}/workstationConfigs', request_field='workstationConfig', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified workstation configuration.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}', http_method='DELETE', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'force', 'validateOnly'], relative_path='v1beta/{+name}', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the requested workstation configuration.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkstationConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}', http_method='GET', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsGetRequest', response_type_name='WorkstationConfig', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}:getIamPolicy', http_method='GET', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1beta/{+resource}:getIamPolicy', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Returns all workstation configurations in the specified cluster.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkstationConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs', http_method='GET', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/workstationConfigs', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsListRequest', response_type_name='ListWorkstationConfigsResponse', supports_download=False)

    def ListUsable(self, request, global_params=None):
        """Returns all workstation configurations in the specified cluster on which the caller has the "workstations.workstation.create" permission.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsListUsableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUsableWorkstationConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('ListUsable')
        return self._RunMethod(config, request, global_params=global_params)
    ListUsable.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs:listUsable', http_method='GET', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.listUsable', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/workstationConfigs:listUsable', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsListUsableRequest', response_type_name='ListUsableWorkstationConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing workstation configuration.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}', http_method='PATCH', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask', 'validateOnly'], relative_path='v1beta/{+name}', request_field='workstationConfig', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}:setIamPolicy', http_method='POST', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}:testIamPermissions', http_method='POST', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)