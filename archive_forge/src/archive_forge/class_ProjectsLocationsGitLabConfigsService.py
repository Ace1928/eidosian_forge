from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsLocationsGitLabConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_gitLabConfigs resource."""
    _NAME = 'projects_locations_gitLabConfigs'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsLocationsGitLabConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `GitLabConfig`. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsGitLabConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gitLabConfigs', http_method='POST', method_id='cloudbuild.projects.locations.gitLabConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['gitlabConfigId'], relative_path='v1/{+parent}/gitLabConfigs', request_field='gitLabConfig', request_type_name='CloudbuildProjectsLocationsGitLabConfigsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a `GitLabConfig`. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsGitLabConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gitLabConfigs/{gitLabConfigsId}', http_method='DELETE', method_id='cloudbuild.projects.locations.gitLabConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsGitLabConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a `GitLabConfig`. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsGitLabConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GitLabConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gitLabConfigs/{gitLabConfigsId}', http_method='GET', method_id='cloudbuild.projects.locations.gitLabConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsGitLabConfigsGetRequest', response_type_name='GitLabConfig', supports_download=False)

    def List(self, request, global_params=None):
        """List all `GitLabConfigs` for a given project. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsGitLabConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGitLabConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gitLabConfigs', http_method='GET', method_id='cloudbuild.projects.locations.gitLabConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/gitLabConfigs', request_field='', request_type_name='CloudbuildProjectsLocationsGitLabConfigsListRequest', response_type_name='ListGitLabConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing `GitLabConfig`. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsGitLabConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gitLabConfigs/{gitLabConfigsId}', http_method='PATCH', method_id='cloudbuild.projects.locations.gitLabConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='gitLabConfig', request_type_name='CloudbuildProjectsLocationsGitLabConfigsPatchRequest', response_type_name='Operation', supports_download=False)

    def RemoveGitLabConnectedRepository(self, request, global_params=None):
        """Remove a GitLab repository from a given GitLabConfig's connected repositories. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsGitLabConfigsRemoveGitLabConnectedRepositoryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('RemoveGitLabConnectedRepository')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveGitLabConnectedRepository.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gitLabConfigs/{gitLabConfigsId}:removeGitLabConnectedRepository', http_method='POST', method_id='cloudbuild.projects.locations.gitLabConfigs.removeGitLabConnectedRepository', ordered_params=['config'], path_params=['config'], query_params=[], relative_path='v1/{+config}:removeGitLabConnectedRepository', request_field='removeGitLabConnectedRepositoryRequest', request_type_name='CloudbuildProjectsLocationsGitLabConfigsRemoveGitLabConnectedRepositoryRequest', response_type_name='Empty', supports_download=False)