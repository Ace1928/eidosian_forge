from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsGithubInstallationsService(base_api.BaseApiService):
    """Service class for the projects_github_installations resource."""
    _NAME = 'projects_github_installations'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsGithubInstallationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create an association between a GCP project and a GitHub installation. This API is experimental.

      Args:
        request: (CloudbuildProjectsGithubInstallationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.github.installations.create', ordered_params=['projectId'], path_params=['projectId'], query_params=['parent', 'projectId1', 'userOauthCode'], relative_path='v1/projects/{projectId}/github/installations', request_field='installation', request_type_name='CloudbuildProjectsGithubInstallationsCreateRequest', response_type_name='Empty', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete an association between a GCP project and a GitHub installation. This API is experimental.

      Args:
        request: (CloudbuildProjectsGithubInstallationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='cloudbuild.projects.github.installations.delete', ordered_params=['projectId', 'installationId'], path_params=['installationId', 'projectId'], query_params=['name'], relative_path='v1/projects/{projectId}/github/installations/{installationId}', request_field='', request_type_name='CloudbuildProjectsGithubInstallationsDeleteRequest', response_type_name='Empty', supports_download=False)

    def List(self, request, global_params=None):
        """List all Installations for a given project id. This API is experimental.

      Args:
        request: (CloudbuildProjectsGithubInstallationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGitHubInstallationsForProjectResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudbuild.projects.github.installations.list', ordered_params=['projectId'], path_params=['projectId'], query_params=['parent'], relative_path='v1/projects/{projectId}/github/installations', request_field='', request_type_name='CloudbuildProjectsGithubInstallationsListRequest', response_type_name='ListGitHubInstallationsForProjectResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update settings for a GCP project to GitHub installation mapping. This API is experimental.

      Args:
        request: (CloudbuildProjectsGithubInstallationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='cloudbuild.projects.github.installations.patch', ordered_params=['projectId', 'id'], path_params=['id', 'projectId'], query_params=['installationId', 'name', 'projectId1', 'updateMask'], relative_path='v1/projects/{projectId}/github/installations/{id}', request_field='installation', request_type_name='CloudbuildProjectsGithubInstallationsPatchRequest', response_type_name='Empty', supports_download=False)