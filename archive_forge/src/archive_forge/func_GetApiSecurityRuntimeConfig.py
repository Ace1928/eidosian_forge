from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetApiSecurityRuntimeConfig(self, request, global_params=None):
    """Gets the API Security runtime configuration for an environment. This named ApiSecurityRuntimeConfig to prevent conflicts with ApiSecurityConfig from addon config.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetApiSecurityRuntimeConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiSecurityRuntimeConfig) The response message.
      """
    config = self.GetMethodConfig('GetApiSecurityRuntimeConfig')
    return self._RunMethod(config, request, global_params=global_params)