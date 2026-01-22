from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.backupdr.v1 import backupdr_v1_messages as messages
class ProjectsLocationsBackupPlanAssociationsService(base_api.BaseApiService):
    """Service class for the projects_locations_backupPlanAssociations resource."""
    _NAME = 'projects_locations_backupPlanAssociations'

    def __init__(self, client):
        super(BackupdrV1.ProjectsLocationsBackupPlanAssociationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a BackupPlanAssociation.

      Args:
        request: (BackupdrProjectsLocationsBackupPlanAssociationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlanAssociations', http_method='POST', method_id='backupdr.projects.locations.backupPlanAssociations.create', ordered_params=['parent'], path_params=['parent'], query_params=['backupPlanAssociationId', 'requestId'], relative_path='v1/{+parent}/backupPlanAssociations', request_field='backupPlanAssociation', request_type_name='BackupdrProjectsLocationsBackupPlanAssociationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single BackupPlanAssociation.

      Args:
        request: (BackupdrProjectsLocationsBackupPlanAssociationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlanAssociations/{backupPlanAssociationsId}', http_method='DELETE', method_id='backupdr.projects.locations.backupPlanAssociations.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='BackupdrProjectsLocationsBackupPlanAssociationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single BackupPlanAssociation.

      Args:
        request: (BackupdrProjectsLocationsBackupPlanAssociationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackupPlanAssociation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlanAssociations/{backupPlanAssociationsId}', http_method='GET', method_id='backupdr.projects.locations.backupPlanAssociations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BackupdrProjectsLocationsBackupPlanAssociationsGetRequest', response_type_name='BackupPlanAssociation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists BackupPlanAssociations in a given project and location.

      Args:
        request: (BackupdrProjectsLocationsBackupPlanAssociationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupPlanAssociationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlanAssociations', http_method='GET', method_id='backupdr.projects.locations.backupPlanAssociations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/backupPlanAssociations', request_field='', request_type_name='BackupdrProjectsLocationsBackupPlanAssociationsListRequest', response_type_name='ListBackupPlanAssociationsResponse', supports_download=False)

    def TriggerBackup(self, request, global_params=None):
        """Triggers a new Backup.

      Args:
        request: (BackupdrProjectsLocationsBackupPlanAssociationsTriggerBackupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('TriggerBackup')
        return self._RunMethod(config, request, global_params=global_params)
    TriggerBackup.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlanAssociations/{backupPlanAssociationsId}:triggerBackup', http_method='POST', method_id='backupdr.projects.locations.backupPlanAssociations.triggerBackup', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:triggerBackup', request_field='triggerBackupRequest', request_type_name='BackupdrProjectsLocationsBackupPlanAssociationsTriggerBackupRequest', response_type_name='Operation', supports_download=False)