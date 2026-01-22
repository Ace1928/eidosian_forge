from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.speech.v2 import speech_v2_messages as messages
class ProjectsLocationsRecognizersService(base_api.BaseApiService):
    """Service class for the projects_locations_recognizers resource."""
    _NAME = 'projects_locations_recognizers'

    def __init__(self, client):
        super(SpeechV2.ProjectsLocationsRecognizersService, self).__init__(client)
        self._upload_configs = {}

    def BatchRecognize(self, request, global_params=None):
        """Performs batch asynchronous speech recognition: send a request with N audio files and receive a long running operation that can be polled to see when the transcriptions are finished.

      Args:
        request: (BatchRecognizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('BatchRecognize')
        return self._RunMethod(config, request, global_params=global_params)
    BatchRecognize.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/recognizers/{recognizersId}:batchRecognize', http_method='POST', method_id='speech.projects.locations.recognizers.batchRecognize', ordered_params=['recognizer'], path_params=['recognizer'], query_params=[], relative_path='v2/{+recognizer}:batchRecognize', request_field='<request>', request_type_name='BatchRecognizeRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a Recognizer.

      Args:
        request: (SpeechProjectsLocationsRecognizersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/recognizers', http_method='POST', method_id='speech.projects.locations.recognizers.create', ordered_params=['parent'], path_params=['parent'], query_params=['recognizerId', 'validateOnly'], relative_path='v2/{+parent}/recognizers', request_field='recognizer', request_type_name='SpeechProjectsLocationsRecognizersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the Recognizer.

      Args:
        request: (SpeechProjectsLocationsRecognizersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/recognizers/{recognizersId}', http_method='DELETE', method_id='speech.projects.locations.recognizers.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'validateOnly'], relative_path='v2/{+name}', request_field='', request_type_name='SpeechProjectsLocationsRecognizersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the requested Recognizer. Fails with NOT_FOUND if the requested Recognizer doesn't exist.

      Args:
        request: (SpeechProjectsLocationsRecognizersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Recognizer) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/recognizers/{recognizersId}', http_method='GET', method_id='speech.projects.locations.recognizers.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SpeechProjectsLocationsRecognizersGetRequest', response_type_name='Recognizer', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Recognizers.

      Args:
        request: (SpeechProjectsLocationsRecognizersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRecognizersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/recognizers', http_method='GET', method_id='speech.projects.locations.recognizers.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v2/{+parent}/recognizers', request_field='', request_type_name='SpeechProjectsLocationsRecognizersListRequest', response_type_name='ListRecognizersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the Recognizer.

      Args:
        request: (SpeechProjectsLocationsRecognizersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/recognizers/{recognizersId}', http_method='PATCH', method_id='speech.projects.locations.recognizers.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v2/{+name}', request_field='recognizer', request_type_name='SpeechProjectsLocationsRecognizersPatchRequest', response_type_name='Operation', supports_download=False)

    def Recognize(self, request, global_params=None):
        """Performs synchronous Speech recognition: receive results after all audio has been sent and processed.

      Args:
        request: (SpeechProjectsLocationsRecognizersRecognizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RecognizeResponse) The response message.
      """
        config = self.GetMethodConfig('Recognize')
        return self._RunMethod(config, request, global_params=global_params)
    Recognize.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/recognizers/{recognizersId}:recognize', http_method='POST', method_id='speech.projects.locations.recognizers.recognize', ordered_params=['recognizer'], path_params=['recognizer'], query_params=[], relative_path='v2/{+recognizer}:recognize', request_field='recognizeRequest', request_type_name='SpeechProjectsLocationsRecognizersRecognizeRequest', response_type_name='RecognizeResponse', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes the Recognizer.

      Args:
        request: (UndeleteRecognizerRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/recognizers/{recognizersId}:undelete', http_method='POST', method_id='speech.projects.locations.recognizers.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:undelete', request_field='<request>', request_type_name='UndeleteRecognizerRequest', response_type_name='Operation', supports_download=False)