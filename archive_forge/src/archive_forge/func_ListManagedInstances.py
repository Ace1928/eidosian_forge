from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListManagedInstances(self, request, global_params=None):
    """Lists the instances in the managed instance group and instances that are scheduled to be created. The list includes any current actions that the group has scheduled for its instances. The orderBy query parameter is not supported. The `pageToken` query parameter is supported only if the group's `listManagedInstancesResults` field is set to `PAGINATED`.

      Args:
        request: (ComputeRegionInstanceGroupManagersListManagedInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionInstanceGroupManagersListInstancesResponse) The response message.
      """
    config = self.GetMethodConfig('ListManagedInstances')
    return self._RunMethod(config, request, global_params=global_params)