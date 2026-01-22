from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
class ProjectsLocationsSourcesMigratingVmsReplicationCyclesService(base_api.BaseApiService):
    """Service class for the projects_locations_sources_migratingVms_replicationCycles resource."""
    _NAME = 'projects_locations_sources_migratingVms_replicationCycles'

    def __init__(self, client):
        super(VmmigrationV1.ProjectsLocationsSourcesMigratingVmsReplicationCyclesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single ReplicationCycle.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsReplicationCyclesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReplicationCycle) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}/replicationCycles/{replicationCyclesId}', http_method='GET', method_id='vmmigration.projects.locations.sources.migratingVms.replicationCycles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsReplicationCyclesGetRequest', response_type_name='ReplicationCycle', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ReplicationCycles in a given MigratingVM.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsReplicationCyclesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListReplicationCyclesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}/replicationCycles', http_method='GET', method_id='vmmigration.projects.locations.sources.migratingVms.replicationCycles.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/replicationCycles', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsReplicationCyclesListRequest', response_type_name='ListReplicationCyclesResponse', supports_download=False)