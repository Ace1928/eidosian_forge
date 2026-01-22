from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationModelsCreateRequest(_messages.Message):
    """A DialogflowProjectsConversationModelsCreateRequest object.

  Fields:
    googleCloudDialogflowV2ConversationModel: A
      GoogleCloudDialogflowV2ConversationModel resource to be passed as the
      request body.
    parent: The project to create conversation model for. Format: `projects/`
  """
    googleCloudDialogflowV2ConversationModel = _messages.MessageField('GoogleCloudDialogflowV2ConversationModel', 1)
    parent = _messages.StringField(2, required=True)