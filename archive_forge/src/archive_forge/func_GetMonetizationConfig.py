from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetMonetizationConfig(self, request, global_params=None):
    """Gets the monetization configuration for the developer.

      Args:
        request: (ApigeeOrganizationsDevelopersGetMonetizationConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperMonetizationConfig) The response message.
      """
    config = self.GetMethodConfig('GetMonetizationConfig')
    return self._RunMethod(config, request, global_params=global_params)