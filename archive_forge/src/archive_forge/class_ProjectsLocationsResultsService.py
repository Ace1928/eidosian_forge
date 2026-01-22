from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v2 import cloudbuild_v2_messages as messages
class ProjectsLocationsResultsService(base_api.BaseApiService):
    """Service class for the projects_locations_results resource."""
    _NAME = 'projects_locations_results'

    def __init__(self, client):
        super(CloudbuildV2.ProjectsLocationsResultsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets Results of a given project and location.

      Args:
        request: (CloudbuildProjectsLocationsResultsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Result) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/results/{resultsId}', http_method='GET', method_id='cloudbuild.projects.locations.results.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsResultsGetRequest', response_type_name='Result', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Results of a given project and location.

      Args:
        request: (CloudbuildProjectsLocationsResultsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListResultsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/results', http_method='GET', method_id='cloudbuild.projects.locations.results.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/results', request_field='', request_type_name='CloudbuildProjectsLocationsResultsListRequest', response_type_name='ListResultsResponse', supports_download=False)