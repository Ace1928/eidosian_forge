from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
def GenerateAwsClusterAgentToken(self, request, global_params=None):
    """Generates an access token for a cluster agent.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersGenerateAwsClusterAgentTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1GenerateAwsClusterAgentTokenResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateAwsClusterAgentToken')
    return self._RunMethod(config, request, global_params=global_params)