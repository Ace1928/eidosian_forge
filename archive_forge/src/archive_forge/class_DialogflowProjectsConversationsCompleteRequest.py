from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationsCompleteRequest(_messages.Message):
    """A DialogflowProjectsConversationsCompleteRequest object.

  Fields:
    googleCloudDialogflowV2CompleteConversationRequest: A
      GoogleCloudDialogflowV2CompleteConversationRequest resource to be passed
      as the request body.
    name: Required. Resource identifier of the conversation to close. Format:
      `projects//locations//conversations/`.
  """
    googleCloudDialogflowV2CompleteConversationRequest = _messages.MessageField('GoogleCloudDialogflowV2CompleteConversationRequest', 1)
    name = _messages.StringField(2, required=True)