from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
def AttachTrust(self, request, global_params=None):
    """Adds an AD trust to a domain.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsAttachTrustRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('AttachTrust')
    return self._RunMethod(config, request, global_params=global_params)