from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
def GenerateAttachedClusterInstallManifest(self, request, global_params=None):
    """Generates the install manifest to be installed on the target cluster.

      Args:
        request: (GkemulticloudProjectsLocationsGenerateAttachedClusterInstallManifestRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1GenerateAttachedClusterInstallManifestResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateAttachedClusterInstallManifest')
    return self._RunMethod(config, request, global_params=global_params)