from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def AccessStageRddGraph(self, request, global_params=None):
    """Obtain RDD operation graph for a Spark Application Stage. Limits the number of clusters returned as part of the graph to 10000.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsAccessStageRddGraphRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSessionSparkApplicationStageRddOperationGraphResponse) The response message.
      """
    config = self.GetMethodConfig('AccessStageRddGraph')
    return self._RunMethod(config, request, global_params=global_params)