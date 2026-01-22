from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2TextInput(_messages.Message):
    """Auxiliary proto messages. Represents the natural language text to be
  processed.

  Fields:
    languageCode: Required. The language of this conversational query. See
      [Language
      Support](https://cloud.google.com/dialogflow/docs/reference/language)
      for a list of the currently supported language codes. Note that queries
      in the same session do not necessarily need to specify the same
      language.
    text: Required. The UTF-8 encoded natural language text to be processed.
      Text length must not exceed 256 characters for virtual agent
      interactions.
  """
    languageCode = _messages.StringField(1)
    text = _messages.StringField(2)