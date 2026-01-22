from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def AggregatedList(self, request, global_params=None):
    """Retrieves an aggregated list of VPN tunnels. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeVpnTunnelsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VpnTunnelAggregatedList) The response message.
      """
    config = self.GetMethodConfig('AggregatedList')
    return self._RunMethod(config, request, global_params=global_params)