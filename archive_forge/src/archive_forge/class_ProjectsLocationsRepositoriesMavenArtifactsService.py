from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesMavenArtifactsService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_mavenArtifacts resource."""
    _NAME = 'projects_locations_repositories_mavenArtifacts'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesMavenArtifactsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a maven artifact.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesMavenArtifactsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MavenArtifact) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/mavenArtifacts/{mavenArtifactsId}', http_method='GET', method_id='artifactregistry.projects.locations.repositories.mavenArtifacts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesMavenArtifactsGetRequest', response_type_name='MavenArtifact', supports_download=False)

    def List(self, request, global_params=None):
        """Lists maven artifacts.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesMavenArtifactsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMavenArtifactsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/mavenArtifacts', http_method='GET', method_id='artifactregistry.projects.locations.repositories.mavenArtifacts.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/mavenArtifacts', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesMavenArtifactsListRequest', response_type_name='ListMavenArtifactsResponse', supports_download=False)