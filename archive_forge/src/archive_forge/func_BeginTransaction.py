from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
def BeginTransaction(self, request, global_params=None):
    """Starts a new transaction.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsBeginTransactionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BeginTransactionResponse) The response message.
      """
    config = self.GetMethodConfig('BeginTransaction')
    return self._RunMethod(config, request, global_params=global_params)