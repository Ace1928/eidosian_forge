from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.netapp.v1 import netapp_v1_messages as messages
class ProjectsLocationsBackupPoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_backupPolicies resource."""
    _NAME = 'projects_locations_backupPolicies'

    def __init__(self, client):
        super(NetappV1.ProjectsLocationsBackupPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates new backup policy.

      Args:
        request: (NetappProjectsLocationsBackupPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPolicies', http_method='POST', method_id='netapp.projects.locations.backupPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['backupPolicyId'], relative_path='v1/{+parent}/backupPolicies', request_field='backupPolicy', request_type_name='NetappProjectsLocationsBackupPoliciesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Warning! This operation will permanently delete the backup policy.

      Args:
        request: (NetappProjectsLocationsBackupPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPolicies/{backupPoliciesId}', http_method='DELETE', method_id='netapp.projects.locations.backupPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetappProjectsLocationsBackupPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the description of the specified backup policy by backup_policy_id.

      Args:
        request: (NetappProjectsLocationsBackupPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackupPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPolicies/{backupPoliciesId}', http_method='GET', method_id='netapp.projects.locations.backupPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetappProjectsLocationsBackupPoliciesGetRequest', response_type_name='BackupPolicy', supports_download=False)

    def List(self, request, global_params=None):
        """Returns list of all available backup policies.

      Args:
        request: (NetappProjectsLocationsBackupPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPolicies', http_method='GET', method_id='netapp.projects.locations.backupPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/backupPolicies', request_field='', request_type_name='NetappProjectsLocationsBackupPoliciesListRequest', response_type_name='ListBackupPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates settings of a specific backup policy.

      Args:
        request: (NetappProjectsLocationsBackupPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPolicies/{backupPoliciesId}', http_method='PATCH', method_id='netapp.projects.locations.backupPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='backupPolicy', request_type_name='NetappProjectsLocationsBackupPoliciesPatchRequest', response_type_name='Operation', supports_download=False)