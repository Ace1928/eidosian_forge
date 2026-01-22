from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datafusion.v1beta1 import datafusion_v1beta1_messages as messages
class ProjectsLocationsVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_versions resource."""
    _NAME = 'projects_locations_versions'

    def __init__(self, client):
        super(DatafusionV1beta1.ProjectsLocationsVersionsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists possible versions for Data Fusion instances in the specified project and location.

      Args:
        request: (DatafusionProjectsLocationsVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAvailableVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/versions', http_method='GET', method_id='datafusion.projects.locations.versions.list', ordered_params=['parent'], path_params=['parent'], query_params=['latestPatchOnly', 'pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/versions', request_field='', request_type_name='DatafusionProjectsLocationsVersionsListRequest', response_type_name='ListAvailableVersionsResponse', supports_download=False)