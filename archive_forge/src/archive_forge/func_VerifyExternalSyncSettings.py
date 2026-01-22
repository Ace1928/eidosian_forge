from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def VerifyExternalSyncSettings(self, request, global_params=None):
    """Verify External primary instance external sync settings.

      Args:
        request: (SqlProjectsInstancesVerifyExternalSyncSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SqlInstancesVerifyExternalSyncSettingsResponse) The response message.
      """
    config = self.GetMethodConfig('VerifyExternalSyncSettings')
    return self._RunMethod(config, request, global_params=global_params)