from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.netapp.v1 import netapp_v1_messages as messages
class ProjectsLocationsBackupVaultsBackupsService(base_api.BaseApiService):
    """Service class for the projects_locations_backupVaults_backups resource."""
    _NAME = 'projects_locations_backupVaults_backups'

    def __init__(self, client):
        super(NetappV1.ProjectsLocationsBackupVaultsBackupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a backup from the volume specified in the request The backup can be created from the given snapshot if specified in the request. If no snapshot specified, there'll be a new snapshot taken to initiate the backup creation.

      Args:
        request: (NetappProjectsLocationsBackupVaultsBackupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}/backups', http_method='POST', method_id='netapp.projects.locations.backupVaults.backups.create', ordered_params=['parent'], path_params=['parent'], query_params=['backupId'], relative_path='v1/{+parent}/backups', request_field='backup', request_type_name='NetappProjectsLocationsBackupVaultsBackupsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Warning! This operation will permanently delete the backup.

      Args:
        request: (NetappProjectsLocationsBackupVaultsBackupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}/backups/{backupsId}', http_method='DELETE', method_id='netapp.projects.locations.backupVaults.backups.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetappProjectsLocationsBackupVaultsBackupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the description of the specified backup.

      Args:
        request: (NetappProjectsLocationsBackupVaultsBackupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Backup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}/backups/{backupsId}', http_method='GET', method_id='netapp.projects.locations.backupVaults.backups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetappProjectsLocationsBackupVaultsBackupsGetRequest', response_type_name='Backup', supports_download=False)

    def List(self, request, global_params=None):
        """Returns descriptions of all backups for a backupVault.

      Args:
        request: (NetappProjectsLocationsBackupVaultsBackupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}/backups', http_method='GET', method_id='netapp.projects.locations.backupVaults.backups.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/backups', request_field='', request_type_name='NetappProjectsLocationsBackupVaultsBackupsListRequest', response_type_name='ListBackupsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update backup with full spec.

      Args:
        request: (NetappProjectsLocationsBackupVaultsBackupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}/backups/{backupsId}', http_method='PATCH', method_id='netapp.projects.locations.backupVaults.backups.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='backup', request_type_name='NetappProjectsLocationsBackupVaultsBackupsPatchRequest', response_type_name='Operation', supports_download=False)