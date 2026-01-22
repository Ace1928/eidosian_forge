from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsLocationsBitbucketServerConfigsConnectedRepositoriesService(base_api.BaseApiService):
    """Service class for the projects_locations_bitbucketServerConfigs_connectedRepositories resource."""
    _NAME = 'projects_locations_bitbucketServerConfigs_connectedRepositories'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsLocationsBitbucketServerConfigsConnectedRepositoriesService, self).__init__(client)
        self._upload_configs = {}

    def BatchCreate(self, request, global_params=None):
        """Batch connecting Bitbucket Server repositories to Cloud Build.

      Args:
        request: (CloudbuildProjectsLocationsBitbucketServerConfigsConnectedRepositoriesBatchCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('BatchCreate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchCreate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}/connectedRepositories:batchCreate', http_method='POST', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.connectedRepositories.batchCreate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/connectedRepositories:batchCreate', request_field='batchCreateBitbucketServerConnectedRepositoriesRequest', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsConnectedRepositoriesBatchCreateRequest', response_type_name='Operation', supports_download=False)