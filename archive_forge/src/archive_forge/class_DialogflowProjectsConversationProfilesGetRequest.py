from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationProfilesGetRequest(_messages.Message):
    """A DialogflowProjectsConversationProfilesGetRequest object.

  Fields:
    name: Required. The resource name of the conversation profile. Format:
      `projects//locations//conversationProfiles/`.
  """
    name = _messages.StringField(1, required=True)