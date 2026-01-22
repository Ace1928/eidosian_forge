from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datapipelines.v1 import datapipelines_v1_messages as messages
class ProjectsLocationsPipelinesJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_pipelines_jobs resource."""
    _NAME = 'projects_locations_pipelines_jobs'

    def __init__(self, client):
        super(DatapipelinesV1.ProjectsLocationsPipelinesJobsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists jobs for a given pipeline. Throws a "FORBIDDEN" error if the caller doesn't have permission to access it.

      Args:
        request: (DatapipelinesProjectsLocationsPipelinesJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatapipelinesV1ListJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelines/{pipelinesId}/jobs', http_method='GET', method_id='datapipelines.projects.locations.pipelines.jobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/jobs', request_field='', request_type_name='DatapipelinesProjectsLocationsPipelinesJobsListRequest', response_type_name='GoogleCloudDatapipelinesV1ListJobsResponse', supports_download=False)