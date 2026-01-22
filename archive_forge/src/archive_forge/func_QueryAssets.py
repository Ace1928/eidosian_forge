from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
def QueryAssets(self, request, global_params=None):
    """Issue a job that queries assets using a SQL statement compatible with [BigQuery SQL](https://cloud.google.com/bigquery/docs/introduction-sql). If the query execution finishes within timeout and there's no pagination, the full query results will be returned in the `QueryAssetsResponse`. Otherwise, full query results can be obtained by issuing extra requests with the `job_reference` from the a previous `QueryAssets` call. Note, the query result has approximately 10 GB limitation enforced by [BigQuery](https://cloud.google.com/bigquery/docs/best-practices-performance-output). Queries return larger results will result in errors.

      Args:
        request: (CloudassetQueryAssetsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryAssetsResponse) The response message.
      """
    config = self.GetMethodConfig('QueryAssets')
    return self._RunMethod(config, request, global_params=global_params)