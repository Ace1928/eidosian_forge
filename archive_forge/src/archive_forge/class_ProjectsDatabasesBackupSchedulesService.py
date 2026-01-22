from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
class ProjectsDatabasesBackupSchedulesService(base_api.BaseApiService):
    """Service class for the projects_databases_backupSchedules resource."""
    _NAME = 'projects_databases_backupSchedules'

    def __init__(self, client):
        super(FirestoreV1.ProjectsDatabasesBackupSchedulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a backup schedule on a database. At most two backup schedules can be configured on a database, one daily backup schedule with retention up to 7 days and one weekly backup schedule with retention up to 14 weeks.

      Args:
        request: (FirestoreProjectsDatabasesBackupSchedulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1BackupSchedule) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/backupSchedules', http_method='POST', method_id='firestore.projects.databases.backupSchedules.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/backupSchedules', request_field='googleFirestoreAdminV1BackupSchedule', request_type_name='FirestoreProjectsDatabasesBackupSchedulesCreateRequest', response_type_name='GoogleFirestoreAdminV1BackupSchedule', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a backup schedule.

      Args:
        request: (FirestoreProjectsDatabasesBackupSchedulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/backupSchedules/{backupSchedulesId}', http_method='DELETE', method_id='firestore.projects.databases.backupSchedules.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='FirestoreProjectsDatabasesBackupSchedulesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a backup schedule.

      Args:
        request: (FirestoreProjectsDatabasesBackupSchedulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1BackupSchedule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/backupSchedules/{backupSchedulesId}', http_method='GET', method_id='firestore.projects.databases.backupSchedules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='FirestoreProjectsDatabasesBackupSchedulesGetRequest', response_type_name='GoogleFirestoreAdminV1BackupSchedule', supports_download=False)

    def List(self, request, global_params=None):
        """List backup schedules.

      Args:
        request: (FirestoreProjectsDatabasesBackupSchedulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1ListBackupSchedulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/backupSchedules', http_method='GET', method_id='firestore.projects.databases.backupSchedules.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/backupSchedules', request_field='', request_type_name='FirestoreProjectsDatabasesBackupSchedulesListRequest', response_type_name='GoogleFirestoreAdminV1ListBackupSchedulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a backup schedule.

      Args:
        request: (FirestoreProjectsDatabasesBackupSchedulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1BackupSchedule) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/backupSchedules/{backupSchedulesId}', http_method='PATCH', method_id='firestore.projects.databases.backupSchedules.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleFirestoreAdminV1BackupSchedule', request_type_name='FirestoreProjectsDatabasesBackupSchedulesPatchRequest', response_type_name='GoogleFirestoreAdminV1BackupSchedule', supports_download=False)