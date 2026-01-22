from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsNotebookRuntimesService(base_api.BaseApiService):
    """Service class for the projects_locations_notebookRuntimes resource."""
    _NAME = 'projects_locations_notebookRuntimes'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsNotebookRuntimesService, self).__init__(client)
        self._upload_configs = {}

    def Assign(self, request, global_params=None):
        """Assigns a NotebookRuntime to a user for a particular Notebook file. This method will either returns an existing assignment or generates a new one.

      Args:
        request: (AiplatformProjectsLocationsNotebookRuntimesAssignRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Assign')
        return self._RunMethod(config, request, global_params=global_params)
    Assign.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/notebookRuntimes:assign', http_method='POST', method_id='aiplatform.projects.locations.notebookRuntimes.assign', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/notebookRuntimes:assign', request_field='googleCloudAiplatformV1AssignNotebookRuntimeRequest', request_type_name='AiplatformProjectsLocationsNotebookRuntimesAssignRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a NotebookRuntime.

      Args:
        request: (AiplatformProjectsLocationsNotebookRuntimesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/notebookRuntimes/{notebookRuntimesId}', http_method='DELETE', method_id='aiplatform.projects.locations.notebookRuntimes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsNotebookRuntimesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a NotebookRuntime.

      Args:
        request: (AiplatformProjectsLocationsNotebookRuntimesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1NotebookRuntime) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/notebookRuntimes/{notebookRuntimesId}', http_method='GET', method_id='aiplatform.projects.locations.notebookRuntimes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsNotebookRuntimesGetRequest', response_type_name='GoogleCloudAiplatformV1NotebookRuntime', supports_download=False)

    def List(self, request, global_params=None):
        """Lists NotebookRuntimes in a Location.

      Args:
        request: (AiplatformProjectsLocationsNotebookRuntimesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListNotebookRuntimesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/notebookRuntimes', http_method='GET', method_id='aiplatform.projects.locations.notebookRuntimes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/notebookRuntimes', request_field='', request_type_name='AiplatformProjectsLocationsNotebookRuntimesListRequest', response_type_name='GoogleCloudAiplatformV1ListNotebookRuntimesResponse', supports_download=False)

    def Start(self, request, global_params=None):
        """Starts a NotebookRuntime.

      Args:
        request: (AiplatformProjectsLocationsNotebookRuntimesStartRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Start')
        return self._RunMethod(config, request, global_params=global_params)
    Start.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/notebookRuntimes/{notebookRuntimesId}:start', http_method='POST', method_id='aiplatform.projects.locations.notebookRuntimes.start', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:start', request_field='googleCloudAiplatformV1StartNotebookRuntimeRequest', request_type_name='AiplatformProjectsLocationsNotebookRuntimesStartRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)