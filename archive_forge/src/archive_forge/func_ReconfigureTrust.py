from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
def ReconfigureTrust(self, request, global_params=None):
    """Updates the DNS conditional forwarder.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsReconfigureTrustRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ReconfigureTrust')
    return self._RunMethod(config, request, global_params=global_params)