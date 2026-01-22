from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
class ProjectsLocationsBackupsService(base_api.BaseApiService):
    """Service class for the projects_locations_backups resource."""
    _NAME = 'projects_locations_backups'

    def __init__(self, client):
        super(FirestoreV1.ProjectsLocationsBackupsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a backup.

      Args:
        request: (FirestoreProjectsLocationsBackupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backups/{backupsId}', http_method='DELETE', method_id='firestore.projects.locations.backups.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='FirestoreProjectsLocationsBackupsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a backup.

      Args:
        request: (FirestoreProjectsLocationsBackupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1Backup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backups/{backupsId}', http_method='GET', method_id='firestore.projects.locations.backups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='FirestoreProjectsLocationsBackupsGetRequest', response_type_name='GoogleFirestoreAdminV1Backup', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the backups.

      Args:
        request: (FirestoreProjectsLocationsBackupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1ListBackupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backups', http_method='GET', method_id='firestore.projects.locations.backups.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/backups', request_field='', request_type_name='FirestoreProjectsLocationsBackupsListRequest', response_type_name='GoogleFirestoreAdminV1ListBackupsResponse', supports_download=False)