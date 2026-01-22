from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkconnectivity.v1beta import networkconnectivity_v1beta_messages as messages
def AcceptSpoke(self, request, global_params=None):
    """Accepts a proposal to attach a Network Connectivity Center spoke to a hub.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsAcceptSpokeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('AcceptSpoke')
    return self._RunMethod(config, request, global_params=global_params)