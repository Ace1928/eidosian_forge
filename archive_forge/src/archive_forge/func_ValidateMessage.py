from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsub.v1 import pubsub_v1_messages as messages
def ValidateMessage(self, request, global_params=None):
    """Validates a message against a schema.

      Args:
        request: (PubsubProjectsSchemasValidateMessageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValidateMessageResponse) The response message.
      """
    config = self.GetMethodConfig('ValidateMessage')
    return self._RunMethod(config, request, global_params=global_params)