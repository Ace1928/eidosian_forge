from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SearchKnowledgeRequest(_messages.Message):
    """The request message for Conversations.SearchKnowledge.

  Fields:
    conversation: The conversation (between human agent and end user) where
      the search request is triggered. Format:
      `projects//locations//conversations/`.
    conversationProfile: Required. The conversation profile used to configure
      the search. Format: `projects//locations//conversationProfiles/`.
    latestMessage: The name of the latest conversation message when the
      request is triggered. Format:
      `projects//locations//conversations//messages/`.
    parent: The parent resource contains the conversation profile Format:
      'projects/' or `projects//locations/`.
    query: Required. The natural language text query for knowledge search.
    sessionId: The ID of the search session. The session_id can be combined
      with Dialogflow V3 Agent ID retrieved from conversation profile or on
      its own to identify a search session. The search history of the same
      session will impact the search result. It's up to the API caller to
      choose an appropriate `Session ID`. It can be a random number or some
      type of session identifiers (preferably hashed). The length must not
      exceed 36 characters.
  """
    conversation = _messages.StringField(1)
    conversationProfile = _messages.StringField(2)
    latestMessage = _messages.StringField(3)
    parent = _messages.StringField(4)
    query = _messages.MessageField('GoogleCloudDialogflowV2TextInput', 5)
    sessionId = _messages.StringField(6)