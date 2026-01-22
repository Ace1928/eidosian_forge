from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
def AllocateLuns(self, request, global_params=None):
    """Allocate Volume's Luns.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesAllocateLunsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('AllocateLuns')
    return self._RunMethod(config, request, global_params=global_params)