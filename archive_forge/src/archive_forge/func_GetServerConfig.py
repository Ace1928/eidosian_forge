from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
def GetServerConfig(self, request, global_params=None):
    """Returns configuration info about the Google Kubernetes Engine service.

      Args:
        request: (ContainerProjectsLocationsGetServerConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServerConfig) The response message.
      """
    config = self.GetMethodConfig('GetServerConfig')
    return self._RunMethod(config, request, global_params=global_params)