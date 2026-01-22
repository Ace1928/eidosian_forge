from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListReferrers(self, request, global_params=None):
    """Retrieves a list of resources that refer to the VM instance specified in the request. For example, if the VM instance is part of a managed or unmanaged instance group, the referrers list includes the instance group. For more information, read Viewing referrers to VM instances.

      Args:
        request: (ComputeInstancesListReferrersRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceListReferrers) The response message.
      """
    config = self.GetMethodConfig('ListReferrers')
    return self._RunMethod(config, request, global_params=global_params)