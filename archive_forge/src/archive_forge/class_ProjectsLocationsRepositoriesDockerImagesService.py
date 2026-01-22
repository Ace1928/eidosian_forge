from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesDockerImagesService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_dockerImages resource."""
    _NAME = 'projects_locations_repositories_dockerImages'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesDockerImagesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a docker image.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesDockerImagesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DockerImage) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/dockerImages/{dockerImagesId}', http_method='GET', method_id='artifactregistry.projects.locations.repositories.dockerImages.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesDockerImagesGetRequest', response_type_name='DockerImage', supports_download=False)

    def List(self, request, global_params=None):
        """Lists docker images.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesDockerImagesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDockerImagesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/dockerImages', http_method='GET', method_id='artifactregistry.projects.locations.repositories.dockerImages.list', ordered_params=['parent'], path_params=['parent'], query_params=['orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/dockerImages', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesDockerImagesListRequest', response_type_name='ListDockerImagesResponse', supports_download=False)