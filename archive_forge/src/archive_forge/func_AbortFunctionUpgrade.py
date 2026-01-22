from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudfunctions.v2 import cloudfunctions_v2_messages as messages
def AbortFunctionUpgrade(self, request, global_params=None):
    """Aborts generation upgrade process for a function with the given name from the specified project. Deletes all 2nd Gen copy related configuration and resources which were created during the upgrade process.

      Args:
        request: (CloudfunctionsProjectsLocationsFunctionsAbortFunctionUpgradeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('AbortFunctionUpgrade')
    return self._RunMethod(config, request, global_params=global_params)