from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3 import translate_v3_messages as messages
def AdaptiveMtTranslate(self, request, global_params=None):
    """Translate text using Adaptive MT.

      Args:
        request: (TranslateProjectsLocationsAdaptiveMtTranslateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AdaptiveMtTranslateResponse) The response message.
      """
    config = self.GetMethodConfig('AdaptiveMtTranslate')
    return self._RunMethod(config, request, global_params=global_params)