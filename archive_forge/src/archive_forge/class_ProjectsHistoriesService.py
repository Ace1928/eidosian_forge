from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
class ProjectsHistoriesService(base_api.BaseApiService):
    """Service class for the projects_histories resource."""
    _NAME = 'projects_histories'

    def __init__(self, client):
        super(ToolresultsV1beta3.ProjectsHistoriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a History. The returned History will have the id set. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to write to project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the containing project does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (History) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='toolresults.projects.histories.create', ordered_params=['projectId'], path_params=['projectId'], query_params=['requestId'], relative_path='toolresults/v1beta3/projects/{projectId}/histories', request_field='history', request_type_name='ToolresultsProjectsHistoriesCreateRequest', response_type_name='History', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a History. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the History does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (History) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.get', ordered_params=['projectId', 'historyId'], path_params=['historyId', 'projectId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}', request_field='', request_type_name='ToolresultsProjectsHistoriesGetRequest', response_type_name='History', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Histories for a given Project. The histories are sorted by modification time in descending order. The history_id key will be used to order the history with the same modification time. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read project - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the History does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListHistoriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.list', ordered_params=['projectId'], path_params=['projectId'], query_params=['filterByName', 'pageSize', 'pageToken'], relative_path='toolresults/v1beta3/projects/{projectId}/histories', request_field='', request_type_name='ToolresultsProjectsHistoriesListRequest', response_type_name='ListHistoriesResponse', supports_download=False)