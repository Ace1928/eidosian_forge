from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.speech.v1 import speech_v1_messages as messages
class SpeechService(base_api.BaseApiService):
    """Service class for the speech resource."""
    _NAME = 'speech'

    def __init__(self, client):
        super(SpeechV1.SpeechService, self).__init__(client)
        self._upload_configs = {}

    def Longrunningrecognize(self, request, global_params=None):
        """Performs asynchronous speech recognition: receive results via the google.longrunning.Operations interface. Returns either an `Operation.error` or an `Operation.response` which contains a `LongRunningRecognizeResponse` message. For more information on asynchronous speech recognition, see the [how-to](https://cloud.google.com/speech-to-text/docs/async-recognize).

      Args:
        request: (LongRunningRecognizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Longrunningrecognize')
        return self._RunMethod(config, request, global_params=global_params)
    Longrunningrecognize.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='speech.speech.longrunningrecognize', ordered_params=[], path_params=[], query_params=[], relative_path='v1/speech:longrunningrecognize', request_field='<request>', request_type_name='LongRunningRecognizeRequest', response_type_name='Operation', supports_download=False)

    def Recognize(self, request, global_params=None):
        """Performs synchronous speech recognition: receive results after all audio has been sent and processed.

      Args:
        request: (RecognizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RecognizeResponse) The response message.
      """
        config = self.GetMethodConfig('Recognize')
        return self._RunMethod(config, request, global_params=global_params)
    Recognize.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='speech.speech.recognize', ordered_params=[], path_params=[], query_params=[], relative_path='v1/speech:recognize', request_field='<request>', request_type_name='RecognizeRequest', response_type_name='RecognizeResponse', supports_download=False)