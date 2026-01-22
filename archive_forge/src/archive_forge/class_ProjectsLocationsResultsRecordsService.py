from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v2 import cloudbuild_v2_messages as messages
class ProjectsLocationsResultsRecordsService(base_api.BaseApiService):
    """Service class for the projects_locations_results_records resource."""
    _NAME = 'projects_locations_results_records'

    def __init__(self, client):
        super(CloudbuildV2.ProjectsLocationsResultsRecordsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets Records of a given project and location.

      Args:
        request: (CloudbuildProjectsLocationsResultsRecordsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Record) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/results/{resultsId}/records/{recordsId}', http_method='GET', method_id='cloudbuild.projects.locations.results.records.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsResultsRecordsGetRequest', response_type_name='Record', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Records of a given project and location.

      Args:
        request: (CloudbuildProjectsLocationsResultsRecordsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRecordsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/results/{resultsId}/records', http_method='GET', method_id='cloudbuild.projects.locations.results.records.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/records', request_field='', request_type_name='CloudbuildProjectsLocationsResultsRecordsListRequest', response_type_name='ListRecordsResponse', supports_download=False)