from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
def UpdateFulfillment(self, request, global_params=None):
    """Updates the fulfillment.

      Args:
        request: (DialogflowProjectsLocationsAgentUpdateFulfillmentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Fulfillment) The response message.
      """
    config = self.GetMethodConfig('UpdateFulfillment')
    return self._RunMethod(config, request, global_params=global_params)