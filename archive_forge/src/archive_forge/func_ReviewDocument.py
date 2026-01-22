from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.documentai.v1 import documentai_v1_messages as messages
def ReviewDocument(self, request, global_params=None):
    """Send a document for Human Review. The input document should be processed by the specified processor.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsHumanReviewConfigReviewDocumentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('ReviewDocument')
    return self._RunMethod(config, request, global_params=global_params)