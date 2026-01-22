from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsConversationsParticipantsSuggestionsSuggestFaqAnswersRequest(_messages.Message):
    """A DialogflowProjectsLocationsConversationsParticipantsSuggestionsSuggest
  FaqAnswersRequest object.

  Fields:
    googleCloudDialogflowV2SuggestFaqAnswersRequest: A
      GoogleCloudDialogflowV2SuggestFaqAnswersRequest resource to be passed as
      the request body.
    parent: Required. The name of the participant to fetch suggestion for.
      Format: `projects//locations//conversations//participants/`.
  """
    googleCloudDialogflowV2SuggestFaqAnswersRequest = _messages.MessageField('GoogleCloudDialogflowV2SuggestFaqAnswersRequest', 1)
    parent = _messages.StringField(2, required=True)