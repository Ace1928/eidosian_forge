from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apphub.v1alpha import apphub_v1alpha_messages as messages
def FindUnregistered(self, request, global_params=None):
    """Finds unregistered workloads in a host project and location.

      Args:
        request: (ApphubProjectsLocationsDiscoveredWorkloadsFindUnregisteredRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FindUnregisteredWorkloadsResponse) The response message.
      """
    config = self.GetMethodConfig('FindUnregistered')
    return self._RunMethod(config, request, global_params=global_params)