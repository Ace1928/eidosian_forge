from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudfunctions.v2 import cloudfunctions_v2_messages as messages
def CommitFunctionUpgrade(self, request, global_params=None):
    """Finalizes the upgrade after which function upgrade can not be rolled back. This is the last step of the multi step process to upgrade 1st Gen functions to 2nd Gen. Deletes all original 1st Gen related configuration and resources.

      Args:
        request: (CloudfunctionsProjectsLocationsFunctionsCommitFunctionUpgradeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('CommitFunctionUpgrade')
    return self._RunMethod(config, request, global_params=global_params)