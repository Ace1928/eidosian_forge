from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsPrivateCloudsUpgradeJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_privateClouds_upgradeJobs resource."""
    _NAME = 'projects_locations_privateClouds_upgradeJobs'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsPrivateCloudsUpgradeJobsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves a Private Cloud `UpgradeJob` resource by its resource name.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsUpgradeJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UpgradeJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/upgradeJobs/{upgradeJobsId}', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.upgradeJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsUpgradeJobsGetRequest', response_type_name='UpgradeJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `UpgradeJob` recources for a given private cloud.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsUpgradeJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUpgradeJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/upgradeJobs', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.upgradeJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/upgradeJobs', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsUpgradeJobsListRequest', response_type_name='ListUpgradeJobsResponse', supports_download=False)