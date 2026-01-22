from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudtasks.v2 import cloudtasks_v2_messages as messages
def GetCmekConfig(self, request, global_params=None):
    """Gets the CMEK config. Gets the Customer Managed Encryption Key configured with the Cloud Tasks lcoation. By default there is no kms_key configured.

      Args:
        request: (CloudtasksProjectsLocationsGetCmekConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CmekConfig) The response message.
      """
    config = self.GetMethodConfig('GetCmekConfig')
    return self._RunMethod(config, request, global_params=global_params)