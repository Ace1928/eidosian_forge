from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
def CreateDocument(self, request, global_params=None):
    """Creates a new document.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsCreateDocumentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Document) The response message.
      """
    config = self.GetMethodConfig('CreateDocument')
    return self._RunMethod(config, request, global_params=global_params)