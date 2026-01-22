from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
def ListIpOwners(self, request, global_params=None):
    """Lists the internal IP owners in the specified network.

      Args:
        request: (ComputeNetworksListIpOwnersRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IpOwnerList) The response message.
      """
    config = self.GetMethodConfig('ListIpOwners')
    return self._RunMethod(config, request, global_params=global_params)