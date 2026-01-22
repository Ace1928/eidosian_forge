from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.backupdr.v1 import backupdr_v1_messages as messages
class ProjectsLocationsBackupVaultsDataSourcesService(base_api.BaseApiService):
    """Service class for the projects_locations_backupVaults_dataSources resource."""
    _NAME = 'projects_locations_backupVaults_dataSources'

    def __init__(self, client):
        super(BackupdrV1.ProjectsLocationsBackupVaultsDataSourcesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single DataSource.

      Args:
        request: (BackupdrProjectsLocationsBackupVaultsDataSourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DataSource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}/dataSources/{dataSourcesId}', http_method='GET', method_id='backupdr.projects.locations.backupVaults.dataSources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BackupdrProjectsLocationsBackupVaultsDataSourcesGetRequest', response_type_name='DataSource', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DataSources in a given project and location.

      Args:
        request: (BackupdrProjectsLocationsBackupVaultsDataSourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDataSourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupVaults/{backupVaultsId}/dataSources', http_method='GET', method_id='backupdr.projects.locations.backupVaults.dataSources.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/dataSources', request_field='', request_type_name='BackupdrProjectsLocationsBackupVaultsDataSourcesListRequest', response_type_name='ListDataSourcesResponse', supports_download=False)