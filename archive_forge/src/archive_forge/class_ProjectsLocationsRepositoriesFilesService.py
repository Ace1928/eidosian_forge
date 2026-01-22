from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesFilesService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_files resource."""
    _NAME = 'projects_locations_repositories_files'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesFilesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a file and all of its content. It is only allowed on generic repositories. The returned operation will complete once the file has been deleted.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesFilesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/files/{filesId}', http_method='DELETE', method_id='artifactregistry.projects.locations.repositories.files.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesFilesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a file.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesFilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleDevtoolsArtifactregistryV1File) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/files/{filesId}', http_method='GET', method_id='artifactregistry.projects.locations.repositories.files.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesFilesGetRequest', response_type_name='GoogleDevtoolsArtifactregistryV1File', supports_download=False)

    def List(self, request, global_params=None):
        """Lists files.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesFilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/files', http_method='GET', method_id='artifactregistry.projects.locations.repositories.files.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/files', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesFilesListRequest', response_type_name='ListFilesResponse', supports_download=False)