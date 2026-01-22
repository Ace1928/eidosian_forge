from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
def RunAggregationQuery(self, request, global_params=None):
    """Runs an aggregation query. Rather than producing Document results like Firestore.RunQuery, this API allows running an aggregation to produce a series of AggregationResult server-side. High-Level Example: ``` -- Return the number of documents in table given a filter. SELECT COUNT(*) FROM ( SELECT * FROM k where a = true ); ```.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsRunAggregationQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RunAggregationQueryResponse) The response message.
      """
    config = self.GetMethodConfig('RunAggregationQuery')
    return self._RunMethod(config, request, global_params=global_params)