from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.stream.v1 import stream_v1_messages as messages
class ProjectsLocationsStreamContentsService(base_api.BaseApiService):
    """Service class for the projects_locations_streamContents resource."""
    _NAME = 'projects_locations_streamContents'

    def __init__(self, client):
        super(StreamV1.ProjectsLocationsStreamContentsService, self).__init__(client)
        self._upload_configs = {}

    def Build(self, request, global_params=None):
        """Builds the content to a Stream compatible format using the associated sources in a consumer cloud storage bucket. A new content version is created with the user-specified tag if the build succeeds. The returned Operation can be used to track the build status by polling operations.get. The Operation will complete when the build is done. Returns [StreamContent] in the Operation.response field on successful completion.

      Args:
        request: (StreamProjectsLocationsStreamContentsBuildRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Build')
        return self._RunMethod(config, request, global_params=global_params)
    Build.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streamContents/{streamContentsId}:build', http_method='POST', method_id='stream.projects.locations.streamContents.build', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:build', request_field='buildStreamContentRequest', request_type_name='StreamProjectsLocationsStreamContentsBuildRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new StreamContent that manages the metadata and builds of user-provided Stream compatible content sources in a consumer cloud storage bucket. The returned Operation can be used to track the creation status by polling operations.get. The Operation will complete when the creation is done. Returns [StreamContent] in the Operation.response field on successful completion.

      Args:
        request: (StreamProjectsLocationsStreamContentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streamContents', http_method='POST', method_id='stream.projects.locations.streamContents.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'streamContentId'], relative_path='v1/{+parent}/streamContents', request_field='streamContent', request_type_name='StreamProjectsLocationsStreamContentsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single StreamContent. This method removes the version history of content builds but does not delete any content source in the consumer cloud storage bucket. The returned Operation can be used to track the deletion status by polling operations.get. The Operation will complete when the deletion is done. Returns Empty in the Operation.response field on successful completion.

      Args:
        request: (StreamProjectsLocationsStreamContentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streamContents/{streamContentsId}', http_method='DELETE', method_id='stream.projects.locations.streamContents.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='StreamProjectsLocationsStreamContentsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single StreamContent.

      Args:
        request: (StreamProjectsLocationsStreamContentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StreamContent) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streamContents/{streamContentsId}', http_method='GET', method_id='stream.projects.locations.streamContents.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='StreamProjectsLocationsStreamContentsGetRequest', response_type_name='StreamContent', supports_download=False)

    def List(self, request, global_params=None):
        """Lists StreamContents in a given project and location.

      Args:
        request: (StreamProjectsLocationsStreamContentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListStreamContentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streamContents', http_method='GET', method_id='stream.projects.locations.streamContents.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/streamContents', request_field='', request_type_name='StreamProjectsLocationsStreamContentsListRequest', response_type_name='ListStreamContentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single StreamContent.

      Args:
        request: (StreamProjectsLocationsStreamContentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streamContents/{streamContentsId}', http_method='PATCH', method_id='stream.projects.locations.streamContents.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='streamContent', request_type_name='StreamProjectsLocationsStreamContentsPatchRequest', response_type_name='Operation', supports_download=False)