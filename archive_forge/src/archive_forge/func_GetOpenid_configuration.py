from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
def GetOpenid_configuration(self, request, global_params=None):
    """Gets the OIDC discovery document for the cluster. See the [OpenID Connect Discovery 1.0 specification](https://openid.net/specs/openid-connect-discovery-1_0.html) for details.

      Args:
        request: (GkemulticloudProjectsLocationsAzureClustersWellKnownGetOpenidConfigurationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AzureOpenIdConfig) The response message.
      """
    config = self.GetMethodConfig('GetOpenid_configuration')
    return self._RunMethod(config, request, global_params=global_params)