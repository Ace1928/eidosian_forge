from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3beta1 import translate_v3beta1_messages as messages
def GetSupportedLanguages(self, request, global_params=None):
    """Returns a list of supported languages for translation.

      Args:
        request: (TranslateProjectsGetSupportedLanguagesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SupportedLanguages) The response message.
      """
    config = self.GetMethodConfig('GetSupportedLanguages')
    return self._RunMethod(config, request, global_params=global_params)