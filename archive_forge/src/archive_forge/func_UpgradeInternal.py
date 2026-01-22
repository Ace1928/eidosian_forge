from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.notebooks.v1 import notebooks_v1_messages as messages
def UpgradeInternal(self, request, global_params=None):
    """Allows notebook instances to call this endpoint to upgrade themselves. Do not use this method directly.

      Args:
        request: (NotebooksProjectsLocationsInstancesUpgradeInternalRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpgradeInternal')
    return self._RunMethod(config, request, global_params=global_params)