from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
class ProjectsHistoriesExecutionsStepsTestCasesService(base_api.BaseApiService):
    """Service class for the projects_histories_executions_steps_testCases resource."""
    _NAME = 'projects_histories_executions_steps_testCases'

    def __init__(self, client):
        super(ToolresultsV1beta3.ProjectsHistoriesExecutionsStepsTestCasesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a Test Case for a Step. Experimental test cases API. Still in active development. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to write to project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the containing Test Case does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsTestCasesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestCase) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.steps.testCases.get', ordered_params=['projectId', 'historyId', 'executionId', 'stepId', 'testCaseId'], path_params=['executionId', 'historyId', 'projectId', 'stepId', 'testCaseId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}/testCases/{testCaseId}', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsTestCasesGetRequest', response_type_name='TestCase', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Test Cases attached to a Step. Experimental test cases API. Still in active development. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to write to project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the containing Step does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsTestCasesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTestCasesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.steps.testCases.list', ordered_params=['projectId', 'historyId', 'executionId', 'stepId'], path_params=['executionId', 'historyId', 'projectId', 'stepId'], query_params=['pageSize', 'pageToken'], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}/testCases', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsTestCasesListRequest', response_type_name='ListTestCasesResponse', supports_download=False)