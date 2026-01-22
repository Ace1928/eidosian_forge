from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ListConversationDatasetsResponse(_messages.Message):
    """The response message for ConversationDatasets.ListConversationDatasets.

  Fields:
    conversationDatasets: The list of datasets to return.
    nextPageToken: The token to use to retrieve the next page of results, or
      empty if there are no more results in the list.
  """
    conversationDatasets = _messages.MessageField('GoogleCloudDialogflowV2ConversationDataset', 1, repeated=True)
    nextPageToken = _messages.StringField(2)