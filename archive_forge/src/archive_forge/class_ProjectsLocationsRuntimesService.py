from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.notebooks.v1 import notebooks_v1_messages as messages
class ProjectsLocationsRuntimesService(base_api.BaseApiService):
    """Service class for the projects_locations_runtimes resource."""
    _NAME = 'projects_locations_runtimes'

    def __init__(self, client):
        super(NotebooksV1.ProjectsLocationsRuntimesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Runtime in a given project and location.

      Args:
        request: (NotebooksProjectsLocationsRuntimesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes', http_method='POST', method_id='notebooks.projects.locations.runtimes.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'runtimeId'], relative_path='v1/{+parent}/runtimes', request_field='runtime', request_type_name='NotebooksProjectsLocationsRuntimesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Runtime.

      Args:
        request: (NotebooksProjectsLocationsRuntimesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}', http_method='DELETE', method_id='notebooks.projects.locations.runtimes.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NotebooksProjectsLocationsRuntimesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Diagnose(self, request, global_params=None):
        """Creates a Diagnostic File and runs Diagnostic Tool given a Runtime.

      Args:
        request: (NotebooksProjectsLocationsRuntimesDiagnoseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Diagnose')
        return self._RunMethod(config, request, global_params=global_params)
    Diagnose.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}:diagnose', http_method='POST', method_id='notebooks.projects.locations.runtimes.diagnose', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:diagnose', request_field='diagnoseRuntimeRequest', request_type_name='NotebooksProjectsLocationsRuntimesDiagnoseRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Runtime. The location must be a regional endpoint rather than zonal.

      Args:
        request: (NotebooksProjectsLocationsRuntimesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Runtime) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}', http_method='GET', method_id='notebooks.projects.locations.runtimes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NotebooksProjectsLocationsRuntimesGetRequest', response_type_name='Runtime', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NotebooksProjectsLocationsRuntimesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}:getIamPolicy', http_method='GET', method_id='notebooks.projects.locations.runtimes.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='NotebooksProjectsLocationsRuntimesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Runtimes in a given project and location.

      Args:
        request: (NotebooksProjectsLocationsRuntimesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRuntimesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes', http_method='GET', method_id='notebooks.projects.locations.runtimes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/runtimes', request_field='', request_type_name='NotebooksProjectsLocationsRuntimesListRequest', response_type_name='ListRuntimesResponse', supports_download=False)

    def Migrate(self, request, global_params=None):
        """Migrate an existing Runtime to a new Workbench Instance.

      Args:
        request: (NotebooksProjectsLocationsRuntimesMigrateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Migrate')
        return self._RunMethod(config, request, global_params=global_params)
    Migrate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}:migrate', http_method='POST', method_id='notebooks.projects.locations.runtimes.migrate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:migrate', request_field='migrateRuntimeRequest', request_type_name='NotebooksProjectsLocationsRuntimesMigrateRequest', response_type_name='Operation', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update Notebook Runtime configuration.

      Args:
        request: (NotebooksProjectsLocationsRuntimesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}', http_method='PATCH', method_id='notebooks.projects.locations.runtimes.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='runtime', request_type_name='NotebooksProjectsLocationsRuntimesPatchRequest', response_type_name='Operation', supports_download=False)

    def RefreshRuntimeTokenInternal(self, request, global_params=None):
        """Gets an access token for the consumer service account that the customer attached to the runtime. Only accessible from the tenant instance.

      Args:
        request: (NotebooksProjectsLocationsRuntimesRefreshRuntimeTokenInternalRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RefreshRuntimeTokenInternalResponse) The response message.
      """
        config = self.GetMethodConfig('RefreshRuntimeTokenInternal')
        return self._RunMethod(config, request, global_params=global_params)
    RefreshRuntimeTokenInternal.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}:refreshRuntimeTokenInternal', http_method='POST', method_id='notebooks.projects.locations.runtimes.refreshRuntimeTokenInternal', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:refreshRuntimeTokenInternal', request_field='refreshRuntimeTokenInternalRequest', request_type_name='NotebooksProjectsLocationsRuntimesRefreshRuntimeTokenInternalRequest', response_type_name='RefreshRuntimeTokenInternalResponse', supports_download=False)

    def ReportEvent(self, request, global_params=None):
        """Reports and processes a runtime event.

      Args:
        request: (NotebooksProjectsLocationsRuntimesReportEventRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ReportEvent')
        return self._RunMethod(config, request, global_params=global_params)
    ReportEvent.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}:reportEvent', http_method='POST', method_id='notebooks.projects.locations.runtimes.reportEvent', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:reportEvent', request_field='reportRuntimeEventRequest', request_type_name='NotebooksProjectsLocationsRuntimesReportEventRequest', response_type_name='Operation', supports_download=False)

    def Reset(self, request, global_params=None):
        """Resets a Managed Notebook Runtime.

      Args:
        request: (NotebooksProjectsLocationsRuntimesResetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Reset')
        return self._RunMethod(config, request, global_params=global_params)
    Reset.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}:reset', http_method='POST', method_id='notebooks.projects.locations.runtimes.reset', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:reset', request_field='resetRuntimeRequest', request_type_name='NotebooksProjectsLocationsRuntimesResetRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NotebooksProjectsLocationsRuntimesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}:setIamPolicy', http_method='POST', method_id='notebooks.projects.locations.runtimes.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='NotebooksProjectsLocationsRuntimesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Start(self, request, global_params=None):
        """Starts a Managed Notebook Runtime. Perform "Start" on GPU instances; "Resume" on CPU instances See: https://cloud.google.com/compute/docs/instances/stop-start-instance https://cloud.google.com/compute/docs/instances/suspend-resume-instance.

      Args:
        request: (NotebooksProjectsLocationsRuntimesStartRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Start')
        return self._RunMethod(config, request, global_params=global_params)
    Start.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}:start', http_method='POST', method_id='notebooks.projects.locations.runtimes.start', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:start', request_field='startRuntimeRequest', request_type_name='NotebooksProjectsLocationsRuntimesStartRequest', response_type_name='Operation', supports_download=False)

    def Stop(self, request, global_params=None):
        """Stops a Managed Notebook Runtime. Perform "Stop" on GPU instances; "Suspend" on CPU instances See: https://cloud.google.com/compute/docs/instances/stop-start-instance https://cloud.google.com/compute/docs/instances/suspend-resume-instance.

      Args:
        request: (NotebooksProjectsLocationsRuntimesStopRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Stop')
        return self._RunMethod(config, request, global_params=global_params)
    Stop.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}:stop', http_method='POST', method_id='notebooks.projects.locations.runtimes.stop', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:stop', request_field='stopRuntimeRequest', request_type_name='NotebooksProjectsLocationsRuntimesStopRequest', response_type_name='Operation', supports_download=False)

    def Switch(self, request, global_params=None):
        """Switch a Managed Notebook Runtime.

      Args:
        request: (NotebooksProjectsLocationsRuntimesSwitchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Switch')
        return self._RunMethod(config, request, global_params=global_params)
    Switch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}:switch', http_method='POST', method_id='notebooks.projects.locations.runtimes.switch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:switch', request_field='switchRuntimeRequest', request_type_name='NotebooksProjectsLocationsRuntimesSwitchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NotebooksProjectsLocationsRuntimesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}:testIamPermissions', http_method='POST', method_id='notebooks.projects.locations.runtimes.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='NotebooksProjectsLocationsRuntimesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Upgrade(self, request, global_params=None):
        """Upgrades a Managed Notebook Runtime to the latest version.

      Args:
        request: (NotebooksProjectsLocationsRuntimesUpgradeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Upgrade')
        return self._RunMethod(config, request, global_params=global_params)
    Upgrade.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/runtimes/{runtimesId}:upgrade', http_method='POST', method_id='notebooks.projects.locations.runtimes.upgrade', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:upgrade', request_field='upgradeRuntimeRequest', request_type_name='NotebooksProjectsLocationsRuntimesUpgradeRequest', response_type_name='Operation', supports_download=False)