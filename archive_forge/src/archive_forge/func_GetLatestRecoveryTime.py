from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def GetLatestRecoveryTime(self, request, global_params=None):
    """Get Latest Recovery Time for a given instance.

      Args:
        request: (SqlProjectsInstancesGetLatestRecoveryTimeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SqlInstancesGetLatestRecoveryTimeResponse) The response message.
      """
    config = self.GetMethodConfig('GetLatestRecoveryTime')
    return self._RunMethod(config, request, global_params=global_params)