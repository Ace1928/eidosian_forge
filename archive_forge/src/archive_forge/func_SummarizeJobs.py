from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def SummarizeJobs(self, request, global_params=None):
    """Obtain summary of Jobs for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSummarizeJobsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SummarizeSessionSparkApplicationJobsResponse) The response message.
      """
    config = self.GetMethodConfig('SummarizeJobs')
    return self._RunMethod(config, request, global_params=global_params)