from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
def ExecuteStreamingSql(self, request, global_params=None):
    """Like ExecuteSql, except returns the result set as a stream. Unlike ExecuteSql, there is no limit on the size of the returned result set. However, no individual row in the result set can exceed 100 MiB, and no column value can exceed 10 MiB.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsExecuteStreamingSqlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PartialResultSet) The response message.
      """
    config = self.GetMethodConfig('ExecuteStreamingSql')
    return self._RunMethod(config, request, global_params=global_params)