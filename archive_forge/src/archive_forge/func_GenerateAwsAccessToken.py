from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
def GenerateAwsAccessToken(self, request, global_params=None):
    """Generates a short-lived access token to authenticate to a given AwsCluster resource.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersGenerateAwsAccessTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1GenerateAwsAccessTokenResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateAwsAccessToken')
    return self._RunMethod(config, request, global_params=global_params)