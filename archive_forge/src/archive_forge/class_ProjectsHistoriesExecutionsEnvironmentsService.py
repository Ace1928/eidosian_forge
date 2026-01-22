from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
class ProjectsHistoriesExecutionsEnvironmentsService(base_api.BaseApiService):
    """Service class for the projects_histories_executions_environments resource."""
    _NAME = 'projects_histories_executions_environments'

    def __init__(self, client):
        super(ToolresultsV1beta3.ProjectsHistoriesExecutionsEnvironmentsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets an Environment. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the Environment does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsEnvironmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Environment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.environments.get', ordered_params=['projectId', 'historyId', 'executionId', 'environmentId'], path_params=['environmentId', 'executionId', 'historyId', 'projectId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/environments/{environmentId}', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsEnvironmentsGetRequest', response_type_name='Environment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Environments for a given Execution. The Environments are sorted by display name. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the containing Execution does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsEnvironmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEnvironmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.environments.list', ordered_params=['projectId', 'historyId', 'executionId'], path_params=['executionId', 'historyId', 'projectId'], query_params=['pageSize', 'pageToken'], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/environments', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsEnvironmentsListRequest', response_type_name='ListEnvironmentsResponse', supports_download=False)