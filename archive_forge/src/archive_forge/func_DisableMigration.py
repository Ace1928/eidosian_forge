from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
def DisableMigration(self, request, global_params=None):
    """Disable Domain Migration.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsDisableMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DisableMigration')
    return self._RunMethod(config, request, global_params=global_params)