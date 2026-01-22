from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
def GetAttachedServerConfig(self, request, global_params=None):
    """Returns information, such as supported Kubernetes versions, on a given Google Cloud location.

      Args:
        request: (GkemulticloudProjectsLocationsGetAttachedServerConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AttachedServerConfig) The response message.
      """
    config = self.GetMethodConfig('GetAttachedServerConfig')
    return self._RunMethod(config, request, global_params=global_params)