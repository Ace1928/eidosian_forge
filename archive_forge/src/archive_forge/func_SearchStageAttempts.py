from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def SearchStageAttempts(self, request, global_params=None):
    """Obtain data corresponding to a spark stage attempts for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsSearchStageAttemptsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchSessionSparkApplicationStageAttemptsResponse) The response message.
      """
    config = self.GetMethodConfig('SearchStageAttempts')
    return self._RunMethod(config, request, global_params=global_params)