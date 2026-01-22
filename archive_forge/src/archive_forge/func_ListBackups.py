from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def ListBackups(self, request, global_params=None):
    """Lists all backups associated with the project.

      Args:
        request: (SqlBackupsListBackupsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupsResponse) The response message.
      """
    config = self.GetMethodConfig('ListBackups')
    return self._RunMethod(config, request, global_params=global_params)