from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
def GetSecuritySettings(self, request, global_params=None):
    """Get Security Settings.

      Args:
        request: (CloudidentityGroupsGetSecuritySettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecuritySettings) The response message.
      """
    config = self.GetMethodConfig('GetSecuritySettings')
    return self._RunMethod(config, request, global_params=global_params)