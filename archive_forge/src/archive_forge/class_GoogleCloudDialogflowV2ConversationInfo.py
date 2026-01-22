from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ConversationInfo(_messages.Message):
    """Represents metadata of a conversation.

  Fields:
    languageCode: Optional. The language code of the conversation data within
      this dataset. See https://cloud.google.com/apis/design/standard_fields
      for more information. Supports all UTF-8 languages.
  """
    languageCode = _messages.StringField(1)