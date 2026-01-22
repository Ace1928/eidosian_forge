from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
def FetchInventory(self, request, global_params=None):
    """List remote source's inventory of VMs. The remote source is the onprem vCenter (remote in the sense it's not in Compute Engine). The inventory describes the list of existing VMs in that source. Note that this operation lists the VMs on the remote source, as opposed to listing the MigratingVms resources in the vmmigration service.

      Args:
        request: (VmmigrationProjectsLocationsSourcesFetchInventoryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchInventoryResponse) The response message.
      """
    config = self.GetMethodConfig('FetchInventory')
    return self._RunMethod(config, request, global_params=global_params)