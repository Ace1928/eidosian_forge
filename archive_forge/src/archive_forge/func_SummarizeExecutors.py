from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def SummarizeExecutors(self, request, global_params=None):
    """Obtain summary of Executor Summary for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSummarizeExecutorsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SummarizeSessionSparkApplicationExecutorsResponse) The response message.
      """
    config = self.GetMethodConfig('SummarizeExecutors')
    return self._RunMethod(config, request, global_params=global_params)