from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
def DisableVpcServiceControls(self, request, global_params=None):
    """Disables VPC service controls for a connection.

      Args:
        request: (ServicenetworkingServicesDisableVpcServiceControlsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DisableVpcServiceControls')
    return self._RunMethod(config, request, global_params=global_params)