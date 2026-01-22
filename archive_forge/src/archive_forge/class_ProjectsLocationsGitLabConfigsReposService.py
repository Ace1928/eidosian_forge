from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsLocationsGitLabConfigsReposService(base_api.BaseApiService):
    """Service class for the projects_locations_gitLabConfigs_repos resource."""
    _NAME = 'projects_locations_gitLabConfigs_repos'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsLocationsGitLabConfigsReposService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List all repositories for a given `GitLabConfig`. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsGitLabConfigsReposListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGitLabRepositoriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gitLabConfigs/{gitLabConfigsId}/repos', http_method='GET', method_id='cloudbuild.projects.locations.gitLabConfigs.repos.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/repos', request_field='', request_type_name='CloudbuildProjectsLocationsGitLabConfigsReposListRequest', response_type_name='ListGitLabRepositoriesResponse', supports_download=False)