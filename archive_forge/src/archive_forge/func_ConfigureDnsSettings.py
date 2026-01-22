from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.domains.v1alpha2 import domains_v1alpha2_messages as messages
def ConfigureDnsSettings(self, request, global_params=None):
    """Updates a `Registration`'s DNS settings.

      Args:
        request: (DomainsProjectsLocationsRegistrationsConfigureDnsSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ConfigureDnsSettings')
    return self._RunMethod(config, request, global_params=global_params)