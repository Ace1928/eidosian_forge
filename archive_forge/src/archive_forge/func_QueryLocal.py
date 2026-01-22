from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
def QueryLocal(self, request, global_params=None):
    """Runs a (possibly multi-step) SQL query asynchronously in the customer project and returns handles that can be used to fetch the results of each step. View references are translated to linked dataset tables, and references to other raw BigQuery tables are permitted.

      Args:
        request: (QueryDataLocalRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryDataResponse) The response message.
      """
    config = self.GetMethodConfig('QueryLocal')
    return self._RunMethod(config, request, global_params=global_params)