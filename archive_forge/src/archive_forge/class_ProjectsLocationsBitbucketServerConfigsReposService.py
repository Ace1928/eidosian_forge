from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsLocationsBitbucketServerConfigsReposService(base_api.BaseApiService):
    """Service class for the projects_locations_bitbucketServerConfigs_repos resource."""
    _NAME = 'projects_locations_bitbucketServerConfigs_repos'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsLocationsBitbucketServerConfigsReposService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List all repositories for a given `BitbucketServerConfig`. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsBitbucketServerConfigsReposListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBitbucketServerRepositoriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}/repos', http_method='GET', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.repos.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/repos', request_field='', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsReposListRequest', response_type_name='ListBitbucketServerRepositoriesResponse', supports_download=False)