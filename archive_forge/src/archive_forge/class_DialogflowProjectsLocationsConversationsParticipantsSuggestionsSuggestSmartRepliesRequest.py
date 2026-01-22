from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsConversationsParticipantsSuggestionsSuggestSmartRepliesRequest(_messages.Message):
    """A DialogflowProjectsLocationsConversationsParticipantsSuggestionsSuggest
  SmartRepliesRequest object.

  Fields:
    googleCloudDialogflowV2SuggestSmartRepliesRequest: A
      GoogleCloudDialogflowV2SuggestSmartRepliesRequest resource to be passed
      as the request body.
    parent: Required. The name of the participant to fetch suggestion for.
      Format: `projects//locations//conversations//participants/`.
  """
    googleCloudDialogflowV2SuggestSmartRepliesRequest = _messages.MessageField('GoogleCloudDialogflowV2SuggestSmartRepliesRequest', 1)
    parent = _messages.StringField(2, required=True)