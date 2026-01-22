from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsConversationsParticipantsPatchRequest(_messages.Message):
    """A DialogflowProjectsLocationsConversationsParticipantsPatchRequest
  object.

  Fields:
    googleCloudDialogflowV2Participant: A GoogleCloudDialogflowV2Participant
      resource to be passed as the request body.
    name: Optional. The unique identifier of this participant. Format:
      `projects//locations//conversations//participants/`.
    updateMask: Required. The mask to specify which fields to update.
  """
    googleCloudDialogflowV2Participant = _messages.MessageField('GoogleCloudDialogflowV2Participant', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)