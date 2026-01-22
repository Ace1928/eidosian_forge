from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationModelsGetRequest(_messages.Message):
    """A DialogflowProjectsConversationModelsGetRequest object.

  Fields:
    name: Required. The conversation model to retrieve. Format:
      `projects//conversationModels/`
  """
    name = _messages.StringField(1, required=True)