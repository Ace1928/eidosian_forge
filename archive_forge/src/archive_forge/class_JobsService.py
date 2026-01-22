from __future__ import absolute_import
from apitools.base.py import base_api
from samples.bigquery_sample.bigquery_v2 import bigquery_v2_messages as messages
class JobsService(base_api.BaseApiService):
    """Service class for the jobs resource."""
    _NAME = u'jobs'

    def __init__(self, client):
        super(BigqueryV2.JobsService, self).__init__(client)
        self._upload_configs = {'Insert': base_api.ApiUploadInfo(accept=['*/*'], max_size=None, resumable_multipart=True, resumable_path=u'/resumable/upload/bigquery/v2/projects/{projectId}/jobs', simple_multipart=True, simple_path=u'/upload/bigquery/v2/projects/{projectId}/jobs')}

    def Cancel(self, request, global_params=None):
        """Requests that a job be cancelled. This call will return immediately, and the client will need to poll for the job status to see if the cancel completed successfully. Cancelled jobs may still incur costs.

      Args:
        request: (BigqueryJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (JobCancelResponse) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'bigquery.jobs.cancel', ordered_params=[u'projectId', u'jobId'], path_params=[u'jobId', u'projectId'], query_params=[], relative_path=u'project/{projectId}/jobs/{jobId}/cancel', request_field='', request_type_name=u'BigqueryJobsCancelRequest', response_type_name=u'JobCancelResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns information about a specific job. Job information is available for a six month period after creation. Requires that you're the person who ran the job, or have the Is Owner project role.

      Args:
        request: (BigqueryJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Job) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'bigquery.jobs.get', ordered_params=[u'projectId', u'jobId'], path_params=[u'jobId', u'projectId'], query_params=[], relative_path=u'projects/{projectId}/jobs/{jobId}', request_field='', request_type_name=u'BigqueryJobsGetRequest', response_type_name=u'Job', supports_download=False)

    def GetQueryResults(self, request, global_params=None):
        """Retrieves the results of a query job.

      Args:
        request: (BigqueryJobsGetQueryResultsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetQueryResultsResponse) The response message.
      """
        config = self.GetMethodConfig('GetQueryResults')
        return self._RunMethod(config, request, global_params=global_params)
    GetQueryResults.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'bigquery.jobs.getQueryResults', ordered_params=[u'projectId', u'jobId'], path_params=[u'jobId', u'projectId'], query_params=[u'maxResults', u'pageToken', u'startIndex', u'timeoutMs'], relative_path=u'projects/{projectId}/queries/{jobId}', request_field='', request_type_name=u'BigqueryJobsGetQueryResultsRequest', response_type_name=u'GetQueryResultsResponse', supports_download=False)

    def Insert(self, request, global_params=None, upload=None):
        """Starts a new asynchronous job. Requires the Can View project role.

      Args:
        request: (BigqueryJobsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        upload: (Upload, default: None) If present, upload
            this stream with the request.
      Returns:
        (Job) The response message.
      """
        config = self.GetMethodConfig('Insert')
        upload_config = self.GetUploadConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params, upload=upload, upload_config=upload_config)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'bigquery.jobs.insert', ordered_params=[u'projectId'], path_params=[u'projectId'], query_params=[], relative_path=u'projects/{projectId}/jobs', request_field=u'job', request_type_name=u'BigqueryJobsInsertRequest', response_type_name=u'Job', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all jobs that you started in the specified project. Job information is available for a six month period after creation. The job list is sorted in reverse chronological order, by job creation time. Requires the Can View project role, or the Is Owner project role if you set the allUsers property.

      Args:
        request: (BigqueryJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (JobList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'bigquery.jobs.list', ordered_params=[u'projectId'], path_params=[u'projectId'], query_params=[u'allUsers', u'maxResults', u'pageToken', u'projection', u'stateFilter'], relative_path=u'projects/{projectId}/jobs', request_field='', request_type_name=u'BigqueryJobsListRequest', response_type_name=u'JobList', supports_download=False)

    def Query(self, request, global_params=None):
        """Runs a BigQuery SQL query synchronously and returns query results if the query completes within a specified timeout.

      Args:
        request: (BigqueryJobsQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryResponse) The response message.
      """
        config = self.GetMethodConfig('Query')
        return self._RunMethod(config, request, global_params=global_params)
    Query.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'bigquery.jobs.query', ordered_params=[u'projectId'], path_params=[u'projectId'], query_params=[], relative_path=u'projects/{projectId}/queries', request_field=u'queryRequest', request_type_name=u'BigqueryJobsQueryRequest', response_type_name=u'QueryResponse', supports_download=False)