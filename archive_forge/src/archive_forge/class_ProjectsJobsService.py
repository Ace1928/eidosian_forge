from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
class ProjectsJobsService(base_api.BaseApiService):
    """Service class for the projects_jobs resource."""
    _NAME = 'projects_jobs'

    def __init__(self, client):
        super(DataflowV1b3.ProjectsJobsService, self).__init__(client)
        self._upload_configs = {}

    def Aggregated(self, request, global_params=None):
        """List the jobs of a project across all regions. **Note:** This method doesn't support filtering the list of jobs by name.

      Args:
        request: (DataflowProjectsJobsAggregatedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListJobsResponse) The response message.
      """
        config = self.GetMethodConfig('Aggregated')
        return self._RunMethod(config, request, global_params=global_params)
    Aggregated.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataflow.projects.jobs.aggregated', ordered_params=['projectId'], path_params=['projectId'], query_params=['filter', 'location', 'name', 'pageSize', 'pageToken', 'view'], relative_path='v1b3/projects/{projectId}/jobs:aggregated', request_field='', request_type_name='DataflowProjectsJobsAggregatedRequest', response_type_name='ListJobsResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a Cloud Dataflow job. To create a job, we recommend using `projects.locations.jobs.create` with a [regional endpoint] (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints). Using `projects.jobs.create` is not recommended, as your job will always start in `us-central1`. Do not enter confidential information when you supply string values using the API.

      Args:
        request: (DataflowProjectsJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Job) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataflow.projects.jobs.create', ordered_params=['projectId'], path_params=['projectId'], query_params=['location', 'replaceJobId', 'view'], relative_path='v1b3/projects/{projectId}/jobs', request_field='job', request_type_name='DataflowProjectsJobsCreateRequest', response_type_name='Job', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the state of the specified Cloud Dataflow job. To get the state of a job, we recommend using `projects.locations.jobs.get` with a [regional endpoint] (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints). Using `projects.jobs.get` is not recommended, as you can only get the state of jobs that are running in `us-central1`.

      Args:
        request: (DataflowProjectsJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Job) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataflow.projects.jobs.get', ordered_params=['projectId', 'jobId'], path_params=['jobId', 'projectId'], query_params=['location', 'view'], relative_path='v1b3/projects/{projectId}/jobs/{jobId}', request_field='', request_type_name='DataflowProjectsJobsGetRequest', response_type_name='Job', supports_download=False)

    def GetMetrics(self, request, global_params=None):
        """Request the job status. To request the status of a job, we recommend using `projects.locations.jobs.getMetrics` with a [regional endpoint] (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints). Using `projects.jobs.getMetrics` is not recommended, as you can only request the status of jobs that are running in `us-central1`.

      Args:
        request: (DataflowProjectsJobsGetMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (JobMetrics) The response message.
      """
        config = self.GetMethodConfig('GetMetrics')
        return self._RunMethod(config, request, global_params=global_params)
    GetMetrics.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataflow.projects.jobs.getMetrics', ordered_params=['projectId', 'jobId'], path_params=['jobId', 'projectId'], query_params=['location', 'startTime'], relative_path='v1b3/projects/{projectId}/jobs/{jobId}/metrics', request_field='', request_type_name='DataflowProjectsJobsGetMetricsRequest', response_type_name='JobMetrics', supports_download=False)

    def List(self, request, global_params=None):
        """List the jobs of a project. To list the jobs of a project in a region, we recommend using `projects.locations.jobs.list` with a [regional endpoint] (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints). To list the all jobs across all regions, use `projects.jobs.aggregated`. Using `projects.jobs.list` is not recommended, because you can only get the list of jobs that are running in `us-central1`. `projects.locations.jobs.list` and `projects.jobs.list` support filtering the list of jobs by name. Filtering by name isn't supported by `projects.jobs.aggregated`.

      Args:
        request: (DataflowProjectsJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataflow.projects.jobs.list', ordered_params=['projectId'], path_params=['projectId'], query_params=['filter', 'location', 'name', 'pageSize', 'pageToken', 'view'], relative_path='v1b3/projects/{projectId}/jobs', request_field='', request_type_name='DataflowProjectsJobsListRequest', response_type_name='ListJobsResponse', supports_download=False)

    def Snapshot(self, request, global_params=None):
        """Snapshot the state of a streaming job.

      Args:
        request: (DataflowProjectsJobsSnapshotRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Snapshot) The response message.
      """
        config = self.GetMethodConfig('Snapshot')
        return self._RunMethod(config, request, global_params=global_params)
    Snapshot.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataflow.projects.jobs.snapshot', ordered_params=['projectId', 'jobId'], path_params=['jobId', 'projectId'], query_params=[], relative_path='v1b3/projects/{projectId}/jobs/{jobId}:snapshot', request_field='snapshotJobRequest', request_type_name='DataflowProjectsJobsSnapshotRequest', response_type_name='Snapshot', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the state of an existing Cloud Dataflow job. To update the state of an existing job, we recommend using `projects.locations.jobs.update` with a [regional endpoint] (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints). Using `projects.jobs.update` is not recommended, as you can only update the state of jobs that are running in `us-central1`.

      Args:
        request: (DataflowProjectsJobsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Job) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='dataflow.projects.jobs.update', ordered_params=['projectId', 'jobId'], path_params=['jobId', 'projectId'], query_params=['location', 'updateMask'], relative_path='v1b3/projects/{projectId}/jobs/{jobId}', request_field='job', request_type_name='DataflowProjectsJobsUpdateRequest', response_type_name='Job', supports_download=False)