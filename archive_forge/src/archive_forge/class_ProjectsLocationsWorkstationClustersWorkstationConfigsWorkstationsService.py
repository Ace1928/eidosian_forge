from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workstations.v1beta import workstations_v1beta_messages as messages
class ProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsService(base_api.BaseApiService):
    """Service class for the projects_locations_workstationClusters_workstationConfigs_workstations resource."""
    _NAME = 'projects_locations_workstationClusters_workstationConfigs_workstations'

    def __init__(self, client):
        super(WorkstationsV1beta.ProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new workstation.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}/workstations', http_method='POST', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.workstations.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly', 'workstationId'], relative_path='v1beta/{+parent}/workstations', request_field='workstation', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified workstation.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}/workstations/{workstationsId}', http_method='DELETE', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.workstations.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'validateOnly'], relative_path='v1beta/{+name}', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def GenerateAccessToken(self, request, global_params=None):
        """Returns a short-lived credential that can be used to send authenticated and authorized traffic to a workstation.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsGenerateAccessTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateAccessTokenResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateAccessToken')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateAccessToken.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}/workstations/{workstationsId}:generateAccessToken', http_method='POST', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.workstations.generateAccessToken', ordered_params=['workstation'], path_params=['workstation'], query_params=[], relative_path='v1beta/{+workstation}:generateAccessToken', request_field='generateAccessTokenRequest', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsGenerateAccessTokenRequest', response_type_name='GenerateAccessTokenResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the requested workstation.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workstation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}/workstations/{workstationsId}', http_method='GET', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.workstations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsGetRequest', response_type_name='Workstation', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}/workstations/{workstationsId}:getIamPolicy', http_method='GET', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.workstations.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1beta/{+resource}:getIamPolicy', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Returns all Workstations using the specified workstation configuration.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkstationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}/workstations', http_method='GET', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.workstations.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/workstations', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsListRequest', response_type_name='ListWorkstationsResponse', supports_download=False)

    def ListUsable(self, request, global_params=None):
        """Returns all workstations using the specified workstation configuration on which the caller has the "workstations.workstations.use" permission.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsListUsableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUsableWorkstationsResponse) The response message.
      """
        config = self.GetMethodConfig('ListUsable')
        return self._RunMethod(config, request, global_params=global_params)
    ListUsable.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}/workstations:listUsable', http_method='GET', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.workstations.listUsable', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/workstations:listUsable', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsListUsableRequest', response_type_name='ListUsableWorkstationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing workstation.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}/workstations/{workstationsId}', http_method='PATCH', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.workstations.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask', 'validateOnly'], relative_path='v1beta/{+name}', request_field='workstation', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}/workstations/{workstationsId}:setIamPolicy', http_method='POST', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.workstations.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Start(self, request, global_params=None):
        """Starts running a workstation so that users can connect to it.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsStartRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Start')
        return self._RunMethod(config, request, global_params=global_params)
    Start.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}/workstations/{workstationsId}:start', http_method='POST', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.workstations.start', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:start', request_field='startWorkstationRequest', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsStartRequest', response_type_name='Operation', supports_download=False)

    def Stop(self, request, global_params=None):
        """Stops running a workstation, reducing costs.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsStopRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Stop')
        return self._RunMethod(config, request, global_params=global_params)
    Stop.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}/workstations/{workstationsId}:stop', http_method='POST', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.workstations.stop', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:stop', request_field='stopWorkstationRequest', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsStopRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}/workstationConfigs/{workstationConfigsId}/workstations/{workstationsId}:testIamPermissions', http_method='POST', method_id='workstations.projects.locations.workstationClusters.workstationConfigs.workstations.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)