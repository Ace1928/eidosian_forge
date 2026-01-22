from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetUrlMap(self, request, global_params=None):
    """Changes the URL map for TargetHttpsProxy.

      Args:
        request: (ComputeTargetHttpsProxiesSetUrlMapRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetUrlMap')
    return self._RunMethod(config, request, global_params=global_params)