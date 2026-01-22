from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def SummarizeStageAttemptTasks(self, request, global_params=None):
    """Obtain summary of Tasks for a Spark Application Stage Attempt.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSummarizeStageAttemptTasksRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SummarizeSessionSparkApplicationStageAttemptTasksResponse) The response message.
      """
    config = self.GetMethodConfig('SummarizeStageAttemptTasks')
    return self._RunMethod(config, request, global_params=global_params)