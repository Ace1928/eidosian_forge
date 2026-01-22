from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
def ExecuteSql(self, request, global_params=None):
    """Executes an SQL statement, returning all results in a single reply. This method cannot be used to return a result set larger than 10 MiB; if the query yields more data than that, the query fails with a `FAILED_PRECONDITION` error. Operations inside read-write transactions might return `ABORTED`. If this occurs, the application should restart the transaction from the beginning. See Transaction for more details. Larger result sets can be fetched in streaming fashion by calling ExecuteStreamingSql instead.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsExecuteSqlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResultSet) The response message.
      """
    config = self.GetMethodConfig('ExecuteSql')
    return self._RunMethod(config, request, global_params=global_params)