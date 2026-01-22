from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def SearchStageAttemptTasks(self, request, global_params=None):
    """Obtain data corresponding to tasks for a spark stage attempt for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchStageAttemptTasksRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationStageAttemptTasksResponse) The response message.
      """
    config = self.GetMethodConfig('SearchStageAttemptTasks')
    return self._RunMethod(config, request, global_params=global_params)