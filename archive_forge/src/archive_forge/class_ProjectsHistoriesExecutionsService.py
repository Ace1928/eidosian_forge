from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
class ProjectsHistoriesExecutionsService(base_api.BaseApiService):
    """Service class for the projects_histories_executions resource."""
    _NAME = 'projects_histories_executions'

    def __init__(self, client):
        super(ToolresultsV1beta3.ProjectsHistoriesExecutionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an Execution. The returned Execution will have the id set. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to write to project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the containing History does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Execution) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='toolresults.projects.histories.executions.create', ordered_params=['projectId', 'historyId'], path_params=['historyId', 'projectId'], query_params=['requestId'], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions', request_field='execution', request_type_name='ToolresultsProjectsHistoriesExecutionsCreateRequest', response_type_name='Execution', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an Execution. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to write to project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the Execution does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Execution) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.get', ordered_params=['projectId', 'historyId', 'executionId'], path_params=['executionId', 'historyId', 'projectId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsGetRequest', response_type_name='Execution', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Executions for a given History. The executions are sorted by creation_time in descending order. The execution_id key will be used to order the executions with the same creation_time. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the containing History does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListExecutionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.list', ordered_params=['projectId', 'historyId'], path_params=['historyId', 'projectId'], query_params=['pageSize', 'pageToken'], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsListRequest', response_type_name='ListExecutionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing Execution with the supplied partial entity. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to write to project - INVALID_ARGUMENT - if the request is malformed - FAILED_PRECONDITION - if the requested state transition is illegal - NOT_FOUND - if the containing History does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Execution) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='toolresults.projects.histories.executions.patch', ordered_params=['projectId', 'historyId', 'executionId'], path_params=['executionId', 'historyId', 'projectId'], query_params=['requestId'], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}', request_field='execution', request_type_name='ToolresultsProjectsHistoriesExecutionsPatchRequest', response_type_name='Execution', supports_download=False)