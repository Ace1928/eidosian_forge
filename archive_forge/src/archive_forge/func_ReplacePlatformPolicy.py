from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.binaryauthorization.v1 import binaryauthorization_v1_messages as messages
def ReplacePlatformPolicy(self, request, global_params=None):
    """Replaces a platform policy. Returns `NOT_FOUND` if the policy doesn't exist.

      Args:
        request: (PlatformPolicy) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PlatformPolicy) The response message.
      """
    config = self.GetMethodConfig('ReplacePlatformPolicy')
    return self._RunMethod(config, request, global_params=global_params)