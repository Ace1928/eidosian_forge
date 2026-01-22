from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.language.v1 import language_v1_messages as messages
def AnalyzeSyntax(self, request, global_params=None):
    """Analyzes the syntax of the text and provides sentence boundaries and tokenization along with part of speech tags, dependency trees, and other properties.

      Args:
        request: (AnalyzeSyntaxRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeSyntaxResponse) The response message.
      """
    config = self.GetMethodConfig('AnalyzeSyntax')
    return self._RunMethod(config, request, global_params=global_params)