from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.memcache.v1beta2 import memcache_v1beta2_messages as messages
def ApplySoftwareUpdate(self, request, global_params=None):
    """Updates software on the selected nodes of the Instance.

      Args:
        request: (MemcacheProjectsLocationsInstancesApplySoftwareUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ApplySoftwareUpdate')
    return self._RunMethod(config, request, global_params=global_params)