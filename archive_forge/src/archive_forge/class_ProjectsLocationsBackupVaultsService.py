from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.backupdr.v1 import backupdr_v1_messages as messages
class ProjectsLocationsBackupVaultsService(base_api.BaseApiService):
    """Service class for the projects_locations_backupVaults resource."""
    _NAME = 'projects_locations_backupVaults'

    def __init__(self, client):
        super(BackupdrV1.ProjectsLocationsBackupVaultsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Backupvault.

      Args:
        request: (BackupdrProjectsLocationsBackupVaultsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults', http_method='POST', method_id='backupdr.projects.locations.backupVaults.create', ordered_params=['parent'], path_params=['parent'], query_params=['backupVaultId', 'requestId', 'validateOnly'], relative_path='v1/{+parent}/backupVaults', request_field='backupVault', request_type_name='BackupdrProjectsLocationsBackupVaultsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single BackupVault.

      Args:
        request: (BackupdrProjectsLocationsBackupVaultsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}', http_method='DELETE', method_id='backupdr.projects.locations.backupVaults.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'force', 'requestId', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='BackupdrProjectsLocationsBackupVaultsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single BackupVault.

      Args:
        request: (BackupdrProjectsLocationsBackupVaultsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackupVault) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}', http_method='GET', method_id='backupdr.projects.locations.backupVaults.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BackupdrProjectsLocationsBackupVaultsGetRequest', response_type_name='BackupVault', supports_download=False)

    def List(self, request, global_params=None):
        """Lists BackupVaults in a given project and location.

      Args:
        request: (BackupdrProjectsLocationsBackupVaultsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupVaultsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults', http_method='GET', method_id='backupdr.projects.locations.backupVaults.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/backupVaults', request_field='', request_type_name='BackupdrProjectsLocationsBackupVaultsListRequest', response_type_name='ListBackupVaultsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single BackupVault.

      Args:
        request: (BackupdrProjectsLocationsBackupVaultsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}', http_method='PATCH', method_id='backupdr.projects.locations.backupVaults.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='backupVault', request_type_name='BackupdrProjectsLocationsBackupVaultsPatchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (BackupdrProjectsLocationsBackupVaultsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}:testIamPermissions', http_method='POST', method_id='backupdr.projects.locations.backupVaults.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='BackupdrProjectsLocationsBackupVaultsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)