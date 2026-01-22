from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
class ProjectsHistoriesExecutionsStepsService(base_api.BaseApiService):
    """Service class for the projects_histories_executions_steps resource."""
    _NAME = 'projects_histories_executions_steps'

    def __init__(self, client):
        super(ToolresultsV1beta3.ProjectsHistoriesExecutionsStepsService, self).__init__(client)
        self._upload_configs = {}

    def AccessibilityClusters(self, request, global_params=None):
        """Lists accessibility clusters for a given Step May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read project - INVALID_ARGUMENT - if the request is malformed - FAILED_PRECONDITION - if an argument in the request happens to be invalid; e.g. if the locale format is incorrect - NOT_FOUND - if the containing Step does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsAccessibilityClustersRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListStepAccessibilityClustersResponse) The response message.
      """
        config = self.GetMethodConfig('AccessibilityClusters')
        return self._RunMethod(config, request, global_params=global_params)
    AccessibilityClusters.method_config = lambda: base_api.ApiMethodInfo(flat_path='toolresults/v1beta3/projects/{projectsId}/histories/{historiesId}/executions/{executionsId}/steps/{stepsId}:accessibilityClusters', http_method='GET', method_id='toolresults.projects.histories.executions.steps.accessibilityClusters', ordered_params=['name'], path_params=['name'], query_params=['locale'], relative_path='toolresults/v1beta3/{+name}:accessibilityClusters', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsAccessibilityClustersRequest', response_type_name='ListStepAccessibilityClustersResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a Step. The returned Step will have the id set. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to write to project - INVALID_ARGUMENT - if the request is malformed - FAILED_PRECONDITION - if the step is too large (more than 10Mib) - NOT_FOUND - if the containing Execution does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Step) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='toolresults.projects.histories.executions.steps.create', ordered_params=['projectId', 'historyId', 'executionId'], path_params=['executionId', 'historyId', 'projectId'], query_params=['requestId'], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps', request_field='step', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsCreateRequest', response_type_name='Step', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Step. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the Step does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Step) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.steps.get', ordered_params=['projectId', 'historyId', 'executionId', 'stepId'], path_params=['executionId', 'historyId', 'projectId', 'stepId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsGetRequest', response_type_name='Step', supports_download=False)

    def GetPerfMetricsSummary(self, request, global_params=None):
        """Retrieves a PerfMetricsSummary. May return any of the following error code(s): - NOT_FOUND - The specified PerfMetricsSummary does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsGetPerfMetricsSummaryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PerfMetricsSummary) The response message.
      """
        config = self.GetMethodConfig('GetPerfMetricsSummary')
        return self._RunMethod(config, request, global_params=global_params)
    GetPerfMetricsSummary.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.steps.getPerfMetricsSummary', ordered_params=['projectId', 'historyId', 'executionId', 'stepId'], path_params=['executionId', 'historyId', 'projectId', 'stepId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}/perfMetricsSummary', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsGetPerfMetricsSummaryRequest', response_type_name='PerfMetricsSummary', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Steps for a given Execution. The steps are sorted by creation_time in descending order. The step_id key will be used to order the steps with the same creation_time. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read project - INVALID_ARGUMENT - if the request is malformed - FAILED_PRECONDITION - if an argument in the request happens to be invalid; e.g. if an attempt is made to list the children of a nonexistent Step - NOT_FOUND - if the containing Execution does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListStepsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.steps.list', ordered_params=['projectId', 'historyId', 'executionId'], path_params=['executionId', 'historyId', 'projectId'], query_params=['pageSize', 'pageToken'], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsListRequest', response_type_name='ListStepsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing Step with the supplied partial entity. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to write project - INVALID_ARGUMENT - if the request is malformed - FAILED_PRECONDITION - if the requested state transition is illegal (e.g try to upload a duplicate xml file), if the updated step is too large (more than 10Mib) - NOT_FOUND - if the containing Execution does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Step) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='toolresults.projects.histories.executions.steps.patch', ordered_params=['projectId', 'historyId', 'executionId', 'stepId'], path_params=['executionId', 'historyId', 'projectId', 'stepId'], query_params=['requestId'], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}', request_field='step', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsPatchRequest', response_type_name='Step', supports_download=False)

    def PublishXunitXmlFiles(self, request, global_params=None):
        """Publish xml files to an existing Step. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to write project - INVALID_ARGUMENT - if the request is malformed - FAILED_PRECONDITION - if the requested state transition is illegal, e.g. try to upload a duplicate xml file or a file too large. - NOT_FOUND - if the containing Execution does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsPublishXunitXmlFilesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Step) The response message.
      """
        config = self.GetMethodConfig('PublishXunitXmlFiles')
        return self._RunMethod(config, request, global_params=global_params)
    PublishXunitXmlFiles.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='toolresults.projects.histories.executions.steps.publishXunitXmlFiles', ordered_params=['projectId', 'historyId', 'executionId', 'stepId'], path_params=['executionId', 'historyId', 'projectId', 'stepId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/steps/{stepId}:publishXunitXmlFiles', request_field='publishXunitXmlFilesRequest', request_type_name='ToolresultsProjectsHistoriesExecutionsStepsPublishXunitXmlFilesRequest', response_type_name='Step', supports_download=False)