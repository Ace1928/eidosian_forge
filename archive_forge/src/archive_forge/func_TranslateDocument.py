from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3beta1 import translate_v3beta1_messages as messages
def TranslateDocument(self, request, global_params=None):
    """Translates documents in synchronous mode.

      Args:
        request: (TranslateProjectsLocationsTranslateDocumentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TranslateDocumentResponse) The response message.
      """
    config = self.GetMethodConfig('TranslateDocument')
    return self._RunMethod(config, request, global_params=global_params)