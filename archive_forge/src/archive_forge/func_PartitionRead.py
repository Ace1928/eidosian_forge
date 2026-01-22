from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
def PartitionRead(self, request, global_params=None):
    """Creates a set of partition tokens that can be used to execute a read operation in parallel. Each of the returned partition tokens can be used by StreamingRead to specify a subset of the read result to read. The same session and read-only transaction must be used by the PartitionReadRequest used to create the partition tokens and the ReadRequests that use the partition tokens. There are no ordering guarantees on rows returned among the returned partition tokens, or even within each individual StreamingRead call issued with a partition_token. Partition tokens become invalid when the session used to create them is deleted, is idle for too long, begins a new transaction, or becomes too old. When any of these happen, it is not possible to resume the read, and the whole operation must be restarted from the beginning.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsPartitionReadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PartitionResponse) The response message.
      """
    config = self.GetMethodConfig('PartitionRead')
    return self._RunMethod(config, request, global_params=global_params)