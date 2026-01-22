from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.speech.v2 import speech_v2_messages as messages
class ProjectsLocationsCustomClassesService(base_api.BaseApiService):
    """Service class for the projects_locations_customClasses resource."""
    _NAME = 'projects_locations_customClasses'

    def __init__(self, client):
        super(SpeechV2.ProjectsLocationsCustomClassesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a CustomClass.

      Args:
        request: (SpeechProjectsLocationsCustomClassesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/customClasses', http_method='POST', method_id='speech.projects.locations.customClasses.create', ordered_params=['parent'], path_params=['parent'], query_params=['customClassId', 'validateOnly'], relative_path='v2/{+parent}/customClasses', request_field='customClass', request_type_name='SpeechProjectsLocationsCustomClassesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the CustomClass.

      Args:
        request: (SpeechProjectsLocationsCustomClassesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/customClasses/{customClassesId}', http_method='DELETE', method_id='speech.projects.locations.customClasses.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'validateOnly'], relative_path='v2/{+name}', request_field='', request_type_name='SpeechProjectsLocationsCustomClassesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the requested CustomClass.

      Args:
        request: (SpeechProjectsLocationsCustomClassesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CustomClass) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/customClasses/{customClassesId}', http_method='GET', method_id='speech.projects.locations.customClasses.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SpeechProjectsLocationsCustomClassesGetRequest', response_type_name='CustomClass', supports_download=False)

    def List(self, request, global_params=None):
        """Lists CustomClasses.

      Args:
        request: (SpeechProjectsLocationsCustomClassesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCustomClassesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/customClasses', http_method='GET', method_id='speech.projects.locations.customClasses.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v2/{+parent}/customClasses', request_field='', request_type_name='SpeechProjectsLocationsCustomClassesListRequest', response_type_name='ListCustomClassesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the CustomClass.

      Args:
        request: (SpeechProjectsLocationsCustomClassesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/customClasses/{customClassesId}', http_method='PATCH', method_id='speech.projects.locations.customClasses.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v2/{+name}', request_field='customClass', request_type_name='SpeechProjectsLocationsCustomClassesPatchRequest', response_type_name='Operation', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes the CustomClass.

      Args:
        request: (UndeleteCustomClassRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/customClasses/{customClassesId}:undelete', http_method='POST', method_id='speech.projects.locations.customClasses.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:undelete', request_field='<request>', request_type_name='UndeleteCustomClassRequest', response_type_name='Operation', supports_download=False)