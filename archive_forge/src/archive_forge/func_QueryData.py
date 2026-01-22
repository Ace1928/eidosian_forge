from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
def QueryData(self, request, global_params=None):
    """Runs a (possibly multi-step) SQL query asynchronously and returns handles that can be used to fetch the results of each step. Raw table references are not permitted; all tables must be referenced in the form of views.

      Args:
        request: (QueryDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryDataResponse) The response message.
      """
    config = self.GetMethodConfig('QueryData')
    return self._RunMethod(config, request, global_params=global_params)