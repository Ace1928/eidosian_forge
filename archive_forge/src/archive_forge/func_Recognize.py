from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.speech.v2 import speech_v2_messages as messages
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