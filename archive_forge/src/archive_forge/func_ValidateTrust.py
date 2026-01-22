from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
def ValidateTrust(self, request, global_params=None):
    """Validates a trust state, that the target domain is reachable, and that the target domain is able to accept incoming trust requests.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsValidateTrustRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ValidateTrust')
    return self._RunMethod(config, request, global_params=global_params)