from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.notebooks.v2 import notebooks_v2_messages as messages
def UpgradeSystem(self, request, global_params=None):
    """Allows notebook instances to upgrade themselves. Do not use this method directly.

      Args:
        request: (NotebooksProjectsLocationsInstancesUpgradeSystemRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpgradeSystem')
    return self._RunMethod(config, request, global_params=global_params)