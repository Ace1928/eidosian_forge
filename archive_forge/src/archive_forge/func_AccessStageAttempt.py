from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def AccessStageAttempt(self, request, global_params=None):
    """Obtain data corresponding to a spark stage attempt for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsAccessStageAttemptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSessionSparkApplicationStageAttemptResponse) The response message.
      """
    config = self.GetMethodConfig('AccessStageAttempt')
    return self._RunMethod(config, request, global_params=global_params)