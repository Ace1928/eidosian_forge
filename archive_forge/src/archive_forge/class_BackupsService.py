from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
class BackupsService(base_api.BaseApiService):
    """Service class for the backups resource."""
    _NAME = 'backups'

    def __init__(self, client):
        super(SqladminV1beta4.BackupsService, self).__init__(client)
        self._upload_configs = {}

    def CreateBackup(self, request, global_params=None):
        """Creates a backup for a cloud sql instance. This API can only be used to create OnDemand backups.

      Args:
        request: (SqlBackupsCreateBackupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CreateBackup')
        return self._RunMethod(config, request, global_params=global_params)
    CreateBackup.method_config = lambda: base_api.ApiMethodInfo(flat_path='sql/v1beta4/projects/{projectsId}/backups', http_method='POST', method_id='sql.backups.createBackup', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='sql/v1beta4/{+parent}/backups', request_field='backup', request_type_name='SqlBackupsCreateBackupRequest', response_type_name='Operation', supports_download=False)

    def DeleteBackup(self, request, global_params=None):
        """Deletes the backup.

      Args:
        request: (SqlBackupsDeleteBackupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DeleteBackup')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteBackup.method_config = lambda: base_api.ApiMethodInfo(flat_path='sql/v1beta4/projects/{projectsId}/backups/{backupsId}', http_method='DELETE', method_id='sql.backups.deleteBackup', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='sql/v1beta4/{+name}', request_field='', request_type_name='SqlBackupsDeleteBackupRequest', response_type_name='Operation', supports_download=False)

    def GetBackup(self, request, global_params=None):
        """Retrieves a resource containing information about a backup.

      Args:
        request: (SqlBackupsGetBackupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Backup) The response message.
      """
        config = self.GetMethodConfig('GetBackup')
        return self._RunMethod(config, request, global_params=global_params)
    GetBackup.method_config = lambda: base_api.ApiMethodInfo(flat_path='sql/v1beta4/projects/{projectsId}/backups/{backupsId}', http_method='GET', method_id='sql.backups.getBackup', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='sql/v1beta4/{+name}', request_field='', request_type_name='SqlBackupsGetBackupRequest', response_type_name='Backup', supports_download=False)

    def ListBackups(self, request, global_params=None):
        """Lists all backups associated with the project.

      Args:
        request: (SqlBackupsListBackupsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupsResponse) The response message.
      """
        config = self.GetMethodConfig('ListBackups')
        return self._RunMethod(config, request, global_params=global_params)
    ListBackups.method_config = lambda: base_api.ApiMethodInfo(flat_path='sql/v1beta4/projects/{projectsId}/backups', http_method='GET', method_id='sql.backups.listBackups', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='sql/v1beta4/{+parent}/backups', request_field='', request_type_name='SqlBackupsListBackupsRequest', response_type_name='ListBackupsResponse', supports_download=False)

    def UpdateBackup(self, request, global_params=None):
        """Updates the retention period and the description of the backup, currently restricted to final backups.

      Args:
        request: (SqlBackupsUpdateBackupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdateBackup')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateBackup.method_config = lambda: base_api.ApiMethodInfo(flat_path='sql/v1beta4/projects/{projectsId}/backups/{backupsId}', http_method='PATCH', method_id='sql.backups.updateBackup', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='sql/v1beta4/{+name}', request_field='backup', request_type_name='SqlBackupsUpdateBackupRequest', response_type_name='Operation', supports_download=False)