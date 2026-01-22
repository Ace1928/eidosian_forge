from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
def ListCollectionIds(self, request, global_params=None):
    """Lists all the collection IDs underneath a document.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsListCollectionIdsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCollectionIdsResponse) The response message.
      """
    config = self.GetMethodConfig('ListCollectionIds')
    return self._RunMethod(config, request, global_params=global_params)