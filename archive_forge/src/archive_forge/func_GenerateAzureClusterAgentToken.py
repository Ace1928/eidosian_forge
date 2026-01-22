from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
def GenerateAzureClusterAgentToken(self, request, global_params=None):
    """Generates an access token for a cluster agent.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClustersGenerateAzureClusterAgentTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1GenerateAzureClusterAgentTokenResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateAzureClusterAgentToken')
    return self._RunMethod(config, request, global_params=global_params)