from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
def GetLdapssettings(self, request, global_params=None):
    """Gets the domain ldaps settings.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsGetLdapssettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LDAPSSettings) The response message.
      """
    config = self.GetMethodConfig('GetLdapssettings')
    return self._RunMethod(config, request, global_params=global_params)