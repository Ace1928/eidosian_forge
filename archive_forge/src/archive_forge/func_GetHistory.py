from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
def GetHistory(self, request, global_params=None):
    """Gets the history of the specified environment.

      Args:
        request: (DialogflowProjectsLocationsAgentEnvironmentsGetHistoryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2EnvironmentHistory) The response message.
      """
    config = self.GetMethodConfig('GetHistory')
    return self._RunMethod(config, request, global_params=global_params)