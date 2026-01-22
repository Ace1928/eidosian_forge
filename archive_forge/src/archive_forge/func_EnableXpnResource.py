from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def EnableXpnResource(self, request, global_params=None):
    """Enable service resource (a.k.a service project) for a host project, so that subnets in the host project can be used by instances in the service project.

      Args:
        request: (ComputeProjectsEnableXpnResourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('EnableXpnResource')
    return self._RunMethod(config, request, global_params=global_params)