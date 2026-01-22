from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
def BatchWrite(self, request, global_params=None):
    """Applies a batch of write operations. The BatchWrite method does not apply the write operations atomically and can apply them out of order. Method does not allow more than one write per document. Each write succeeds or fails independently. See the BatchWriteResponse for the success status of each write. If you require an atomically applied set of writes, use Commit instead.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsBatchWriteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchWriteResponse) The response message.
      """
    config = self.GetMethodConfig('BatchWrite')
    return self._RunMethod(config, request, global_params=global_params)