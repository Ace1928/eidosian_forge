from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories resource."""
    _NAME = 'projects_locations_repositories'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a repository. The returned Operation will finish once the repository has been created. Its response will be the created Repository.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories', http_method='POST', method_id='artifactregistry.projects.locations.repositories.create', ordered_params=['parent'], path_params=['parent'], query_params=['repositoryId'], relative_path='v1/{+parent}/repositories', request_field='repository', request_type_name='ArtifactregistryProjectsLocationsRepositoriesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a repository and all of its contents. The returned Operation will finish once the repository has been deleted. It will not have any Operation metadata and will return a google.protobuf.Empty response.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}', http_method='DELETE', method_id='artifactregistry.projects.locations.repositories.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a repository.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Repository) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}', http_method='GET', method_id='artifactregistry.projects.locations.repositories.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesGetRequest', response_type_name='Repository', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM policy for a given resource.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}:getIamPolicy', http_method='GET', method_id='artifactregistry.projects.locations.repositories.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists repositories.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRepositoriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories', http_method='GET', method_id='artifactregistry.projects.locations.repositories.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/repositories', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesListRequest', response_type_name='ListRepositoriesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a repository.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Repository) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}', http_method='PATCH', method_id='artifactregistry.projects.locations.repositories.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='repository', request_type_name='ArtifactregistryProjectsLocationsRepositoriesPatchRequest', response_type_name='Repository', supports_download=False)

    def Reindex(self, request, global_params=None):
        """Updates the index files for an OS repository. Intended for use on remote repositories to check if the upstream has been updated, and if so pull the new index files.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesReindexRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Reindex')
        return self._RunMethod(config, request, global_params=global_params)
    Reindex.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}:reindex', http_method='POST', method_id='artifactregistry.projects.locations.repositories.reindex', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:reindex', request_field='reindexRepositoryRequest', request_type_name='ArtifactregistryProjectsLocationsRepositoriesReindexRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Updates the IAM policy for a given resource.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}:setIamPolicy', http_method='POST', method_id='artifactregistry.projects.locations.repositories.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ArtifactregistryProjectsLocationsRepositoriesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Tests if the caller has a list of permissions on a resource.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}:testIamPermissions', http_method='POST', method_id='artifactregistry.projects.locations.repositories.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ArtifactregistryProjectsLocationsRepositoriesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)