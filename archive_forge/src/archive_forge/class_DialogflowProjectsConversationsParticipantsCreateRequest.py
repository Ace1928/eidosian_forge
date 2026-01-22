from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationsParticipantsCreateRequest(_messages.Message):
    """A DialogflowProjectsConversationsParticipantsCreateRequest object.

  Fields:
    googleCloudDialogflowV2Participant: A GoogleCloudDialogflowV2Participant
      resource to be passed as the request body.
    parent: Required. Resource identifier of the conversation adding the
      participant. Format: `projects//locations//conversations/`.
  """
    googleCloudDialogflowV2Participant = _messages.MessageField('GoogleCloudDialogflowV2Participant', 1)
    parent = _messages.StringField(2, required=True)