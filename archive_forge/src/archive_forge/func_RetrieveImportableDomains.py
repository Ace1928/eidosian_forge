from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.domains.v1alpha2 import domains_v1alpha2_messages as messages
def RetrieveImportableDomains(self, request, global_params=None):
    """Deprecated: For more information, see [Cloud Domains feature deprecation](https://cloud.google.com/domains/docs/deprecations/feature-deprecations) Lists domain names from [Google Domains](https://domains.google/) that can be imported to Cloud Domains using the `ImportDomain` method. Since individual users can own domains in Google Domains, the list of domains returned depends on the individual user making the call. Domains already managed by Cloud Domains are not returned.

      Args:
        request: (DomainsProjectsLocationsRegistrationsRetrieveImportableDomainsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RetrieveImportableDomainsResponse) The response message.
      """
    config = self.GetMethodConfig('RetrieveImportableDomains')
    return self._RunMethod(config, request, global_params=global_params)