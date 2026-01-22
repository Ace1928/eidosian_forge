from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkconnectivity.v1beta import networkconnectivity_v1beta_messages as messages
class ProjectsLocationsCustomHardwareLinkAttachmentsService(base_api.BaseApiService):
    """Service class for the projects_locations_customHardwareLinkAttachments resource."""
    _NAME = 'projects_locations_customHardwareLinkAttachments'

    def __init__(self, client):
        super(NetworkconnectivityV1beta.ProjectsLocationsCustomHardwareLinkAttachmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new CustomHardwareLinkAttachment in a given project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/customHardwareLinkAttachments', http_method='POST', method_id='networkconnectivity.projects.locations.customHardwareLinkAttachments.create', ordered_params=['parent'], path_params=['parent'], query_params=['customHardwareLinkAttachmentId', 'requestId'], relative_path='v1beta/{+parent}/customHardwareLinkAttachments', request_field='googleCloudNetworkconnectivityV1betaCustomHardwareLinkAttachment', request_type_name='NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single CustomHardwareLinkAttachment.

      Args:
        request: (NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/customHardwareLinkAttachments/{customHardwareLinkAttachmentsId}', http_method='DELETE', method_id='networkconnectivity.projects.locations.customHardwareLinkAttachments.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1beta/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single CustomHardwareLinkAttachment.

      Args:
        request: (NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaCustomHardwareLinkAttachment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/customHardwareLinkAttachments/{customHardwareLinkAttachmentsId}', http_method='GET', method_id='networkconnectivity.projects.locations.customHardwareLinkAttachments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsGetRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaCustomHardwareLinkAttachment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists CustomHardwareLinkAttachments in a given project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaListCustomHardwareLinkAttachmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/customHardwareLinkAttachments', http_method='GET', method_id='networkconnectivity.projects.locations.customHardwareLinkAttachments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/customHardwareLinkAttachments', request_field='', request_type_name='NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsListRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaListCustomHardwareLinkAttachmentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single CustomHardwareLinkAttachment.

      Args:
        request: (NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/customHardwareLinkAttachments/{customHardwareLinkAttachmentsId}', http_method='PATCH', method_id='networkconnectivity.projects.locations.customHardwareLinkAttachments.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1beta/{+name}', request_field='googleCloudNetworkconnectivityV1betaCustomHardwareLinkAttachment', request_type_name='NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)