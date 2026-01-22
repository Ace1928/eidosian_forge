from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
def ValidateLocal(self, request, global_params=None):
    """Validates a query before passing it to QueryDataLocal and returns query metadata synchronously.

      Args:
        request: (QueryDataLocalRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValidateQueryResponse) The response message.
      """
    config = self.GetMethodConfig('ValidateLocal')
    return self._RunMethod(config, request, global_params=global_params)