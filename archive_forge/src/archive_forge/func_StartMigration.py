from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
def StartMigration(self, request, global_params=None):
    """Starts migration for a VM. Starts the process of uploading data and creating snapshots, in replication cycles scheduled by the policy.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsStartMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('StartMigration')
    return self._RunMethod(config, request, global_params=global_params)