from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
def SuggestSmartReplies(self, request, global_params=None):
    """Gets smart replies for a participant based on specific historical messages.

      Args:
        request: (DialogflowProjectsLocationsConversationsParticipantsSuggestionsSuggestSmartRepliesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SuggestSmartRepliesResponse) The response message.
      """
    config = self.GetMethodConfig('SuggestSmartReplies')
    return self._RunMethod(config, request, global_params=global_params)