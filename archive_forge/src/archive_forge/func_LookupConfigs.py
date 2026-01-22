from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha1 import osconfig_v1alpha1_messages as messages
def LookupConfigs(self, request, global_params=None):
    """Lookup the configs that are assigned to an instance. This lookup.
will merge all configs that are assigned to the instance resolving
conflicts as necessary.
This is usually called by the agent running on the instance but can be
called directly to see what configs
This

      Args:
        request: (OsconfigProjectsZonesInstancesLookupConfigsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LookupConfigsResponse) The response message.
      """
    config = self.GetMethodConfig('LookupConfigs')
    return self._RunMethod(config, request, global_params=global_params)