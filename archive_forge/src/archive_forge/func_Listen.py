from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
def Listen(self, request, global_params=None):
    """Listens to changes. This method is only available via gRPC or WebChannel (not REST).

      Args:
        request: (FirestoreProjectsDatabasesDocumentsListenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListenResponse) The response message.
      """
    config = self.GetMethodConfig('Listen')
    return self._RunMethod(config, request, global_params=global_params)