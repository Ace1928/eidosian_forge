from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddeploy.v1 import clouddeploy_v1_messages as messages
def Abandon(self, request, global_params=None):
    """Abandons a Release in the Delivery Pipeline.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesAbandonRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AbandonReleaseResponse) The response message.
      """
    config = self.GetMethodConfig('Abandon')
    return self._RunMethod(config, request, global_params=global_params)