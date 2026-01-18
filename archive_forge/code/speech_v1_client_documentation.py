from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.speech.v1 import speech_v1_messages as messages
Performs synchronous speech recognition: receive results after all audio has been sent and processed.

      Args:
        request: (RecognizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RecognizeResponse) The response message.
      