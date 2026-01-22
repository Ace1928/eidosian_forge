from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetXpnHost(self, request, global_params=None):
    """Gets the shared VPC host project that this project links to. May be empty if no link exists.

      Args:
        request: (ComputeProjectsGetXpnHostRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Project) The response message.
      """
    config = self.GetMethodConfig('GetXpnHost')
    return self._RunMethod(config, request, global_params=global_params)