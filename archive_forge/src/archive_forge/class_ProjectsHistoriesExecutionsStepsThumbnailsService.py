from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
class ProjectsHistoriesExecutionsStepsThumbnailsService(base_api.BaseApiService):
    """Service class for the projects_histories_executions_steps_thumbnails resource."""
    _NAME = 'projects_histories_executions_steps_thumbnails'

    def __init__(self, client):
        super(ToolresultsV1beta3.ProjectsHistoriesExecutionsStepsThumbnailsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists thumbnails of images attached to a step. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read from the project, or from any of the images - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the step does not exist, or if any of the images do not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsThumbnailsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListStepThumbnailsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.steps.thumbnails.list', ordered_params=['projectId', 'historyId', 'executionId', 'stepId'], path_params=['executionId', 'historyId', 'projectId', 'stepId'], query_params=['pageSize', 'pageToken'], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}/thumbnails', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsThumbnailsListRequest', response_type_name='ListStepThumbnailsResponse', supports_download=False)