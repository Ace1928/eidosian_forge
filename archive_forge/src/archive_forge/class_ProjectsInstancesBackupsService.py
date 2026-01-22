from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ProjectsInstancesBackupsService(base_api.BaseApiService):
    """Service class for the projects_instances_backups resource."""
    _NAME = 'projects_instances_backups'

    def __init__(self, client):
        super(SpannerV1.ProjectsInstancesBackupsService, self).__init__(client)
        self._upload_configs = {}

    def Copy(self, request, global_params=None):
        """Starts copying a Cloud Spanner Backup. The returned backup long-running operation will have a name of the format `projects//instances//backups//operations/` and can be used to track copying of the backup. The operation is associated with the destination backup. The metadata field type is CopyBackupMetadata. The response field type is Backup, if successful. Cancelling the returned operation will stop the copying and delete the destination backup. Concurrent CopyBackup requests can run on the same source backup.

      Args:
        request: (SpannerProjectsInstancesBackupsCopyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Copy')
        return self._RunMethod(config, request, global_params=global_params)
    Copy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/backups:copy', http_method='POST', method_id='spanner.projects.instances.backups.copy', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/backups:copy', request_field='copyBackupRequest', request_type_name='SpannerProjectsInstancesBackupsCopyRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Starts creating a new Cloud Spanner Backup. The returned backup long-running operation will have a name of the format `projects//instances//backups//operations/` and can be used to track creation of the backup. The metadata field type is CreateBackupMetadata. The response field type is Backup, if successful. Cancelling the returned operation will stop the creation and delete the backup. There can be only one pending backup creation per database. Backup creation of different databases can run concurrently.

      Args:
        request: (SpannerProjectsInstancesBackupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/backups', http_method='POST', method_id='spanner.projects.instances.backups.create', ordered_params=['parent'], path_params=['parent'], query_params=['backupId', 'encryptionConfig_encryptionType', 'encryptionConfig_kmsKeyName', 'encryptionConfig_kmsKeyNames'], relative_path='v1/{+parent}/backups', request_field='backup', request_type_name='SpannerProjectsInstancesBackupsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a pending or completed Backup.

      Args:
        request: (SpannerProjectsInstancesBackupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/backups/{backupsId}', http_method='DELETE', method_id='spanner.projects.instances.backups.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstancesBackupsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets metadata on a pending or completed Backup.

      Args:
        request: (SpannerProjectsInstancesBackupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Backup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/backups/{backupsId}', http_method='GET', method_id='spanner.projects.instances.backups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstancesBackupsGetRequest', response_type_name='Backup', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a database or backup resource. Returns an empty policy if a database or backup exists but does not have a policy set. Authorization requires `spanner.databases.getIamPolicy` permission on resource. For backups, authorization requires `spanner.backups.getIamPolicy` permission on resource.

      Args:
        request: (SpannerProjectsInstancesBackupsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/backups/{backupsId}:getIamPolicy', http_method='POST', method_id='spanner.projects.instances.backups.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='SpannerProjectsInstancesBackupsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists completed and pending backups. Backups returned are ordered by `create_time` in descending order, starting from the most recent `create_time`.

      Args:
        request: (SpannerProjectsInstancesBackupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/backups', http_method='GET', method_id='spanner.projects.instances.backups.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/backups', request_field='', request_type_name='SpannerProjectsInstancesBackupsListRequest', response_type_name='ListBackupsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a pending or completed Backup.

      Args:
        request: (SpannerProjectsInstancesBackupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Backup) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/backups/{backupsId}', http_method='PATCH', method_id='spanner.projects.instances.backups.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='backup', request_type_name='SpannerProjectsInstancesBackupsPatchRequest', response_type_name='Backup', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on a database or backup resource. Replaces any existing policy. Authorization requires `spanner.databases.setIamPolicy` permission on resource. For backups, authorization requires `spanner.backups.setIamPolicy` permission on resource.

      Args:
        request: (SpannerProjectsInstancesBackupsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/backups/{backupsId}:setIamPolicy', http_method='POST', method_id='spanner.projects.instances.backups.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='SpannerProjectsInstancesBackupsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that the caller has on the specified database or backup resource. Attempting this RPC on a non-existent Cloud Spanner database will result in a NOT_FOUND error if the user has `spanner.databases.list` permission on the containing Cloud Spanner instance. Otherwise returns an empty set of permissions. Calling this method on a backup that does not exist will result in a NOT_FOUND error if the user has `spanner.backups.list` permission on the containing instance.

      Args:
        request: (SpannerProjectsInstancesBackupsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/backups/{backupsId}:testIamPermissions', http_method='POST', method_id='spanner.projects.instances.backups.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='SpannerProjectsInstancesBackupsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)