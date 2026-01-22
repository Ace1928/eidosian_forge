from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.backupdr.v1 import backupdr_v1_messages as messages
class ProjectsLocationsBackupVaultsDataSourcesBackupsService(base_api.BaseApiService):
    """Service class for the projects_locations_backupVaults_dataSources_backups resource."""
    _NAME = 'projects_locations_backupVaults_dataSources_backups'

    def __init__(self, client):
        super(BackupdrV1.ProjectsLocationsBackupVaultsDataSourcesBackupsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a Backup.

      Args:
        request: (BackupdrProjectsLocationsBackupVaultsDataSourcesBackupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}/dataSources/{dataSourcesId}/backups/{backupsId}', http_method='DELETE', method_id='backupdr.projects.locations.backupVaults.dataSources.backups.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='BackupdrProjectsLocationsBackupVaultsDataSourcesBackupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a Backup.

      Args:
        request: (BackupdrProjectsLocationsBackupVaultsDataSourcesBackupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Backup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}/dataSources/{dataSourcesId}/backups/{backupsId}', http_method='GET', method_id='backupdr.projects.locations.backupVaults.dataSources.backups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BackupdrProjectsLocationsBackupVaultsDataSourcesBackupsGetRequest', response_type_name='Backup', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Backups in a given project and location.

      Args:
        request: (BackupdrProjectsLocationsBackupVaultsDataSourcesBackupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}/dataSources/{dataSourcesId}/backups', http_method='GET', method_id='backupdr.projects.locations.backupVaults.dataSources.backups.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/backups', request_field='', request_type_name='BackupdrProjectsLocationsBackupVaultsDataSourcesBackupsListRequest', response_type_name='ListBackupsResponse', supports_download=False)

    def Restore(self, request, global_params=None):
        """Restore Backup.

      Args:
        request: (BackupdrProjectsLocationsBackupVaultsDataSourcesBackupsRestoreRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Restore')
        return self._RunMethod(config, request, global_params=global_params)
    Restore.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}/dataSources/{dataSourcesId}/backups/{backupsId}:restore', http_method='POST', method_id='backupdr.projects.locations.backupVaults.dataSources.backups.restore', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:restore', request_field='restoreBackupRequest', request_type_name='BackupdrProjectsLocationsBackupVaultsDataSourcesBackupsRestoreRequest', response_type_name='Operation', supports_download=False)