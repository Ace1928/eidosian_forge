from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.backupdr.v1 import backupdr_v1_messages as messages
def TriggerBackup(self, request, global_params=None):
    """Triggers a new Backup.

      Args:
        request: (BackupdrProjectsLocationsBackupPlanAssociationsTriggerBackupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('TriggerBackup')
    return self._RunMethod(config, request, global_params=global_params)