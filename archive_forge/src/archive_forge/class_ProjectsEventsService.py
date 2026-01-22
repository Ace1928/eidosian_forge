from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouderrorreporting.v1beta1 import clouderrorreporting_v1beta1_messages as messages
class ProjectsEventsService(base_api.BaseApiService):
    """Service class for the projects_events resource."""
    _NAME = 'projects_events'

    def __init__(self, client):
        super(ClouderrorreportingV1beta1.ProjectsEventsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the specified events.

      Args:
        request: (ClouderrorreportingProjectsEventsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEventsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/events', http_method='GET', method_id='clouderrorreporting.projects.events.list', ordered_params=['projectName'], path_params=['projectName'], query_params=['groupId', 'pageSize', 'pageToken', 'serviceFilter_resourceType', 'serviceFilter_service', 'serviceFilter_version', 'timeRange_period'], relative_path='v1beta1/{+projectName}/events', request_field='', request_type_name='ClouderrorreportingProjectsEventsListRequest', response_type_name='ListEventsResponse', supports_download=False)

    def Report(self, request, global_params=None):
        """Report an individual error event and record the event to a log. This endpoint accepts **either** an OAuth token, **or** an [API key](https://support.google.com/cloud/answer/6158862) for authentication. To use an API key, append it to the URL as the value of a `key` parameter. For example: `POST https://clouderrorreporting.googleapis.com/v1beta1/{projectName}/events:report?key=123ABC456` **Note:** [Error Reporting] (https://cloud.google.com/error-reporting) is a global service built on Cloud Logging and can analyze log entries when all of the following are true: * The log entries are stored in a log bucket in the `global` location. * Customer-managed encryption keys (CMEK) are disabled on the log bucket. * The log bucket satisfies one of the following: * The log bucket is stored in the same project where the logs originated. * The logs were routed to a project, and then that project stored those logs in a log bucket that it owns.

      Args:
        request: (ClouderrorreportingProjectsEventsReportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReportErrorEventResponse) The response message.
      """
        config = self.GetMethodConfig('Report')
        return self._RunMethod(config, request, global_params=global_params)
    Report.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/events:report', http_method='POST', method_id='clouderrorreporting.projects.events.report', ordered_params=['projectName'], path_params=['projectName'], query_params=[], relative_path='v1beta1/{+projectName}/events:report', request_field='reportedErrorEvent', request_type_name='ClouderrorreportingProjectsEventsReportRequest', response_type_name='ReportErrorEventResponse', supports_download=False)