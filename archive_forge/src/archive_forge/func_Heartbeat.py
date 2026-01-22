from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vpcaccess.v1alpha1 import vpcaccess_v1alpha1_messages as messages
def Heartbeat(self, request, global_params=None):
    """A heartbeat from a VM, reporting its IP and serving status.

      Args:
        request: (VpcaccessProjectsLocationsConnectorsHeartbeatRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HeartbeatConnectorResponse) The response message.
      """
    config = self.GetMethodConfig('Heartbeat')
    return self._RunMethod(config, request, global_params=global_params)