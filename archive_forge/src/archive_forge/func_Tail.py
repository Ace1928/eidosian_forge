from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
def Tail(self, request, global_params=None):
    """Streaming read of log entries as they are received. Until the stream is terminated, it will continue reading logs.

      Args:
        request: (TailLogEntriesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TailLogEntriesResponse) The response message.
      """
    config = self.GetMethodConfig('Tail')
    return self._RunMethod(config, request, global_params=global_params)