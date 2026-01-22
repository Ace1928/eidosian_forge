from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkebackup.v1 import gkebackup_v1_messages as messages
def GetBackupIndexDownloadUrl(self, request, global_params=None):
    """Retrieve the link to the backupIndex.

      Args:
        request: (GkebackupProjectsLocationsBackupPlansBackupsGetBackupIndexDownloadUrlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetBackupIndexDownloadUrlResponse) The response message.
      """
    config = self.GetMethodConfig('GetBackupIndexDownloadUrl')
    return self._RunMethod(config, request, global_params=global_params)