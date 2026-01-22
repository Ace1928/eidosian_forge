from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesYumArtifactsService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_yumArtifacts resource."""
    _NAME = 'projects_locations_repositories_yumArtifacts'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesYumArtifactsService, self).__init__(client)
        self._upload_configs = {'Upload': base_api.ApiUploadInfo(accept=['*/*'], max_size=None, resumable_multipart=None, resumable_path=None, simple_multipart=True, simple_path='/upload/v1/{+parent}/yumArtifacts:create')}

    def Import(self, request, global_params=None):
        """Imports Yum (RPM) artifacts. The returned Operation will complete once the resources are imported. Package, Version, and File resources are created based on the imported artifacts. Imported artifacts that conflict with existing resources are ignored.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesYumArtifactsImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/yumArtifacts:import', http_method='POST', method_id='artifactregistry.projects.locations.repositories.yumArtifacts.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/yumArtifacts:import', request_field='importYumArtifactsRequest', request_type_name='ArtifactregistryProjectsLocationsRepositoriesYumArtifactsImportRequest', response_type_name='Operation', supports_download=False)

    def Upload(self, request, global_params=None, upload=None):
        """Directly uploads a Yum artifact. The returned Operation will complete once the resources are uploaded. Package, Version, and File resources are created based on the imported artifact. Imported artifacts that conflict with existing resources are ignored.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesYumArtifactsUploadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        upload: (Upload, default: None) If present, upload
            this stream with the request.
      Returns:
        (UploadYumArtifactMediaResponse) The response message.
      """
        config = self.GetMethodConfig('Upload')
        upload_config = self.GetUploadConfig('Upload')
        return self._RunMethod(config, request, global_params=global_params, upload=upload, upload_config=upload_config)
    Upload.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/yumArtifacts:create', http_method='POST', method_id='artifactregistry.projects.locations.repositories.yumArtifacts.upload', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/yumArtifacts:create', request_field='uploadYumArtifactRequest', request_type_name='ArtifactregistryProjectsLocationsRepositoriesYumArtifactsUploadRequest', response_type_name='UploadYumArtifactMediaResponse', supports_download=False)