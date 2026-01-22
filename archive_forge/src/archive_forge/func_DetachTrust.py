from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
def DetachTrust(self, request, global_params=None):
    """Removes an AD trust.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsDetachTrustRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DetachTrust')
    return self._RunMethod(config, request, global_params=global_params)