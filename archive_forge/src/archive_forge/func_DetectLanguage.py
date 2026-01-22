from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3beta1 import translate_v3beta1_messages as messages
def DetectLanguage(self, request, global_params=None):
    """Detects the language of text within a request.

      Args:
        request: (TranslateProjectsDetectLanguageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DetectLanguageResponse) The response message.
      """
    config = self.GetMethodConfig('DetectLanguage')
    return self._RunMethod(config, request, global_params=global_params)