from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datamigration.v1alpha2 import datamigration_v1alpha2_messages as messages
class ProjectsLocationsMigrationJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_migrationJobs resource."""
    _NAME = 'projects_locations_migrationJobs'

    def __init__(self, client):
        super(DatamigrationV1alpha2.ProjectsLocationsMigrationJobsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new migration job in a given project and location.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs', http_method='POST', method_id='datamigration.projects.locations.migrationJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=['migrationJobId', 'requestId'], relative_path='v1alpha2/{+parent}/migrationJobs', request_field='migrationJob', request_type_name='DatamigrationProjectsLocationsMigrationJobsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single migration job.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}', http_method='DELETE', method_id='datamigration.projects.locations.migrationJobs.delete', ordered_params=['name'], path_params=['name'], query_params=['force', 'requestId'], relative_path='v1alpha2/{+name}', request_field='', request_type_name='DatamigrationProjectsLocationsMigrationJobsDeleteRequest', response_type_name='Operation', supports_download=False)

    def GenerateSshScript(self, request, global_params=None):
        """Generate a SSH configuration script to configure the reverse SSH connectivity.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsGenerateSshScriptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SshScript) The response message.
      """
        config = self.GetMethodConfig('GenerateSshScript')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateSshScript.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}:generateSshScript', http_method='POST', method_id='datamigration.projects.locations.migrationJobs.generateSshScript', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:generateSshScript', request_field='generateSshScriptRequest', request_type_name='DatamigrationProjectsLocationsMigrationJobsGenerateSshScriptRequest', response_type_name='SshScript', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single migration job.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MigrationJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}', http_method='GET', method_id='datamigration.projects.locations.migrationJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='DatamigrationProjectsLocationsMigrationJobsGetRequest', response_type_name='MigrationJob', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}:getIamPolicy', http_method='GET', method_id='datamigration.projects.locations.migrationJobs.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha2/{+resource}:getIamPolicy', request_field='', request_type_name='DatamigrationProjectsLocationsMigrationJobsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists migration jobs in a given project and location.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMigrationJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs', http_method='GET', method_id='datamigration.projects.locations.migrationJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/migrationJobs', request_field='', request_type_name='DatamigrationProjectsLocationsMigrationJobsListRequest', response_type_name='ListMigrationJobsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single migration job.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}', http_method='PATCH', method_id='datamigration.projects.locations.migrationJobs.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha2/{+name}', request_field='migrationJob', request_type_name='DatamigrationProjectsLocationsMigrationJobsPatchRequest', response_type_name='Operation', supports_download=False)

    def Promote(self, request, global_params=None):
        """Promote a migration job, stopping replication to the destination and promoting the destination to be a standalone database.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsPromoteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Promote')
        return self._RunMethod(config, request, global_params=global_params)
    Promote.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}:promote', http_method='POST', method_id='datamigration.projects.locations.migrationJobs.promote', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:promote', request_field='promoteMigrationJobRequest', request_type_name='DatamigrationProjectsLocationsMigrationJobsPromoteRequest', response_type_name='Operation', supports_download=False)

    def Restart(self, request, global_params=None):
        """Restart a stopped or failed migration job, resetting the destination instance to its original state and starting the migration process from scratch.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsRestartRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Restart')
        return self._RunMethod(config, request, global_params=global_params)
    Restart.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}:restart', http_method='POST', method_id='datamigration.projects.locations.migrationJobs.restart', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:restart', request_field='restartMigrationJobRequest', request_type_name='DatamigrationProjectsLocationsMigrationJobsRestartRequest', response_type_name='Operation', supports_download=False)

    def Resume(self, request, global_params=None):
        """Resume a migration job that is currently stopped and is resumable (was stopped during CDC phase).

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsResumeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Resume')
        return self._RunMethod(config, request, global_params=global_params)
    Resume.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}:resume', http_method='POST', method_id='datamigration.projects.locations.migrationJobs.resume', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:resume', request_field='resumeMigrationJobRequest', request_type_name='DatamigrationProjectsLocationsMigrationJobsResumeRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}:setIamPolicy', http_method='POST', method_id='datamigration.projects.locations.migrationJobs.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DatamigrationProjectsLocationsMigrationJobsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Start(self, request, global_params=None):
        """Start an already created migration job.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsStartRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Start')
        return self._RunMethod(config, request, global_params=global_params)
    Start.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}:start', http_method='POST', method_id='datamigration.projects.locations.migrationJobs.start', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:start', request_field='startMigrationJobRequest', request_type_name='DatamigrationProjectsLocationsMigrationJobsStartRequest', response_type_name='Operation', supports_download=False)

    def Stop(self, request, global_params=None):
        """Stops a running migration job.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsStopRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Stop')
        return self._RunMethod(config, request, global_params=global_params)
    Stop.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}:stop', http_method='POST', method_id='datamigration.projects.locations.migrationJobs.stop', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:stop', request_field='stopMigrationJobRequest', request_type_name='DatamigrationProjectsLocationsMigrationJobsStopRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}:testIamPermissions', http_method='POST', method_id='datamigration.projects.locations.migrationJobs.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DatamigrationProjectsLocationsMigrationJobsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Verify(self, request, global_params=None):
        """Verify a migration job, making sure the destination can reach the source and that all configuration and prerequisites are met.

      Args:
        request: (DatamigrationProjectsLocationsMigrationJobsVerifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Verify')
        return self._RunMethod(config, request, global_params=global_params)
    Verify.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/migrationJobs/{migrationJobsId}:verify', http_method='POST', method_id='datamigration.projects.locations.migrationJobs.verify', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:verify', request_field='verifyMigrationJobRequest', request_type_name='DatamigrationProjectsLocationsMigrationJobsVerifyRequest', response_type_name='Operation', supports_download=False)