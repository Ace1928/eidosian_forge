from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
def Master(self, request, global_params=None):
    """Updates the master for a specific cluster.

      Args:
        request: (UpdateMasterRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Master')
    return self._RunMethod(config, request, global_params=global_params)