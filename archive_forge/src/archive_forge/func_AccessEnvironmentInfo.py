from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
def AccessEnvironmentInfo(self, request, global_params=None):
    """Obtain environment details for a Spark Application.

      Args:
        request: (DataprocProjectsLocationsSessionsSparkApplicationsAccessEnvironmentInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessSessionSparkApplicationEnvironmentInfoResponse) The response message.
      """
    config = self.GetMethodConfig('AccessEnvironmentInfo')
    return self._RunMethod(config, request, global_params=global_params)