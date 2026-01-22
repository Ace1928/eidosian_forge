from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class GithubInstallationsProjectsService(base_api.BaseApiService):
    """Service class for the github_installations_projects resource."""
    _NAME = 'github_installations_projects'

    def __init__(self, client):
        super(CloudbuildV1.GithubInstallationsProjectsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """For given installation id, list project-installation mappings across all GCB projects visible to the caller. This API is experimental.

      Args:
        request: (CloudbuildGithubInstallationsProjectsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGitHubInstallationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudbuild.github.installations.projects.list', ordered_params=['installationId'], path_params=['installationId'], query_params=[], relative_path='v1/github/installations/{installationId}/projects', request_field='', request_type_name='CloudbuildGithubInstallationsProjectsListRequest', response_type_name='ListGitHubInstallationsResponse', supports_download=False)