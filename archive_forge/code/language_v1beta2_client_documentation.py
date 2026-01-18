from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.language.v1beta2 import language_v1beta2_messages as messages
Classifies a document into categories.

      Args:
        request: (ClassifyTextRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ClassifyTextResponse) The response message.
      