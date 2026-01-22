from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.language.v1 import language_v1_messages as messages
def AnnotateText(self, request, global_params=None):
    """A convenience method that provides all the features that analyzeSentiment, analyzeEntities, and analyzeSyntax provide in one call.

      Args:
        request: (AnnotateTextRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnnotateTextResponse) The response message.
      """
    config = self.GetMethodConfig('AnnotateText')
    return self._RunMethod(config, request, global_params=global_params)