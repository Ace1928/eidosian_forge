from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsLocationsBitbucketServerConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_bitbucketServerConfigs resource."""
    _NAME = 'projects_locations_bitbucketServerConfigs'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsLocationsBitbucketServerConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `BitbucketServerConfig`. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsBitbucketServerConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs', http_method='POST', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['bitbucketServerConfigId'], relative_path='v1/{+parent}/bitbucketServerConfigs', request_field='bitbucketServerConfig', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a `BitbucketServerConfig`. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsBitbucketServerConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}', http_method='DELETE', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve a `BitbucketServerConfig`. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsBitbucketServerConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BitbucketServerConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}', http_method='GET', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsGetRequest', response_type_name='BitbucketServerConfig', supports_download=False)

    def List(self, request, global_params=None):
        """List all `BitbucketServerConfigs` for a given project. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsBitbucketServerConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBitbucketServerConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs', http_method='GET', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/bitbucketServerConfigs', request_field='', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsListRequest', response_type_name='ListBitbucketServerConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing `BitbucketServerConfig`. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsBitbucketServerConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}', http_method='PATCH', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='bitbucketServerConfig', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsPatchRequest', response_type_name='Operation', supports_download=False)

    def RemoveBitbucketServerConnectedRepository(self, request, global_params=None):
        """Remove a Bitbucket Server repository from a given BitbucketServerConfig's connected repositories. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsBitbucketServerConfigsRemoveBitbucketServerConnectedRepositoryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('RemoveBitbucketServerConnectedRepository')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveBitbucketServerConnectedRepository.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bitbucketServerConfigs/{bitbucketServerConfigsId}:removeBitbucketServerConnectedRepository', http_method='POST', method_id='cloudbuild.projects.locations.bitbucketServerConfigs.removeBitbucketServerConnectedRepository', ordered_params=['config'], path_params=['config'], query_params=[], relative_path='v1/{+config}:removeBitbucketServerConnectedRepository', request_field='removeBitbucketServerConnectedRepositoryRequest', request_type_name='CloudbuildProjectsLocationsBitbucketServerConfigsRemoveBitbucketServerConnectedRepositoryRequest', response_type_name='Empty', supports_download=False)