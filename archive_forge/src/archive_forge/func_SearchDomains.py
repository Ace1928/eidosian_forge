from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.domains.v1alpha2 import domains_v1alpha2_messages as messages
def SearchDomains(self, request, global_params=None):
    """Searches for available domain names similar to the provided query. Availability results from this method are approximate; call `RetrieveRegisterParameters` on a domain before registering to confirm availability.

      Args:
        request: (DomainsProjectsLocationsRegistrationsSearchDomainsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchDomainsResponse) The response message.
      """
    config = self.GetMethodConfig('SearchDomains')
    return self._RunMethod(config, request, global_params=global_params)