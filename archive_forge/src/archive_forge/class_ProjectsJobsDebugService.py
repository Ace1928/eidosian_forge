from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
class ProjectsJobsDebugService(base_api.BaseApiService):
    """Service class for the projects_jobs_debug resource."""
    _NAME = 'projects_jobs_debug'

    def __init__(self, client):
        super(DataflowV1b3.ProjectsJobsDebugService, self).__init__(client)
        self._upload_configs = {}

    def GetConfig(self, request, global_params=None):
        """Get encoded debug configuration for component. Not cacheable.

      Args:
        request: (DataflowProjectsJobsDebugGetConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetDebugConfigResponse) The response message.
      """
        config = self.GetMethodConfig('GetConfig')
        return self._RunMethod(config, request, global_params=global_params)
    GetConfig.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataflow.projects.jobs.debug.getConfig', ordered_params=['projectId', 'jobId'], path_params=['jobId', 'projectId'], query_params=[], relative_path='v1b3/projects/{projectId}/jobs/{jobId}/debug/getConfig', request_field='getDebugConfigRequest', request_type_name='DataflowProjectsJobsDebugGetConfigRequest', response_type_name='GetDebugConfigResponse', supports_download=False)

    def SendCapture(self, request, global_params=None):
        """Send encoded debug capture data for component.

      Args:
        request: (DataflowProjectsJobsDebugSendCaptureRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SendDebugCaptureResponse) The response message.
      """
        config = self.GetMethodConfig('SendCapture')
        return self._RunMethod(config, request, global_params=global_params)
    SendCapture.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataflow.projects.jobs.debug.sendCapture', ordered_params=['projectId', 'jobId'], path_params=['jobId', 'projectId'], query_params=[], relative_path='v1b3/projects/{projectId}/jobs/{jobId}/debug/sendCapture', request_field='sendDebugCaptureRequest', request_type_name='DataflowProjectsJobsDebugSendCaptureRequest', response_type_name='SendDebugCaptureResponse', supports_download=False)