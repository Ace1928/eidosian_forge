from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
def AddGroupMigration(self, request, global_params=None):
    """Adds a MigratingVm to a Group.

      Args:
        request: (VmmigrationProjectsLocationsGroupsAddGroupMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('AddGroupMigration')
    return self._RunMethod(config, request, global_params=global_params)