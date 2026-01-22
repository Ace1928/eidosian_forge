from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
class ProjectsLocationsDataScansJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_dataScans_jobs resource."""
    _NAME = 'projects_locations_dataScans_jobs'

    def __init__(self, client):
        super(DataplexV1.ProjectsLocationsDataScansJobsService, self).__init__(client)
        self._upload_configs = {}

    def GenerateDataQualityRules(self, request, global_params=None):
        """Generates recommended DataQualityRule from a data profiling DataScan.

      Args:
        request: (DataplexProjectsLocationsDataScansJobsGenerateDataQualityRulesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1GenerateDataQualityRulesResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateDataQualityRules')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateDataQualityRules.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans/{dataScansId}/jobs/{jobsId}:generateDataQualityRules', http_method='POST', method_id='dataplex.projects.locations.dataScans.jobs.generateDataQualityRules', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:generateDataQualityRules', request_field='googleCloudDataplexV1GenerateDataQualityRulesRequest', request_type_name='DataplexProjectsLocationsDataScansJobsGenerateDataQualityRulesRequest', response_type_name='GoogleCloudDataplexV1GenerateDataQualityRulesResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a DataScanJob resource.

      Args:
        request: (DataplexProjectsLocationsDataScansJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1DataScanJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans/{dataScansId}/jobs/{jobsId}', http_method='GET', method_id='dataplex.projects.locations.dataScans.jobs.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsDataScansJobsGetRequest', response_type_name='GoogleCloudDataplexV1DataScanJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DataScanJobs under the given DataScan.

      Args:
        request: (DataplexProjectsLocationsDataScansJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1ListDataScanJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataScans/{dataScansId}/jobs', http_method='GET', method_id='dataplex.projects.locations.dataScans.jobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/jobs', request_field='', request_type_name='DataplexProjectsLocationsDataScansJobsListRequest', response_type_name='GoogleCloudDataplexV1ListDataScanJobsResponse', supports_download=False)