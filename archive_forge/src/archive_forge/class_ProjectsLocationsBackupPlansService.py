from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.backupdr.v1 import backupdr_v1_messages as messages
class ProjectsLocationsBackupPlansService(base_api.BaseApiService):
    """Service class for the projects_locations_backupPlans resource."""
    _NAME = 'projects_locations_backupPlans'

    def __init__(self, client):
        super(BackupdrV1.ProjectsLocationsBackupPlansService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a BackupPlan.

      Args:
        request: (BackupdrProjectsLocationsBackupPlansCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans', http_method='POST', method_id='backupdr.projects.locations.backupPlans.create', ordered_params=['parent'], path_params=['parent'], query_params=['backupPlanId', 'requestId'], relative_path='v1/{+parent}/backupPlans', request_field='backupPlan', request_type_name='BackupdrProjectsLocationsBackupPlansCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single BackupPlan.

      Args:
        request: (BackupdrProjectsLocationsBackupPlansDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans/{backupPlansId}', http_method='DELETE', method_id='backupdr.projects.locations.backupPlans.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='BackupdrProjectsLocationsBackupPlansDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single BackupPlan.

      Args:
        request: (BackupdrProjectsLocationsBackupPlansGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackupPlan) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans/{backupPlansId}', http_method='GET', method_id='backupdr.projects.locations.backupPlans.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BackupdrProjectsLocationsBackupPlansGetRequest', response_type_name='BackupPlan', supports_download=False)

    def List(self, request, global_params=None):
        """Lists BackupPlans in a given project and location.

      Args:
        request: (BackupdrProjectsLocationsBackupPlansListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupPlansResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans', http_method='GET', method_id='backupdr.projects.locations.backupPlans.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/backupPlans', request_field='', request_type_name='BackupdrProjectsLocationsBackupPlansListRequest', response_type_name='ListBackupPlansResponse', supports_download=False)