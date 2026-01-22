from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
def ResetAdminPassword(self, request, global_params=None):
    """Resets a domain's administrator password.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsResetAdminPasswordRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResetAdminPasswordResponse) The response message.
      """
    config = self.GetMethodConfig('ResetAdminPassword')
    return self._RunMethod(config, request, global_params=global_params)