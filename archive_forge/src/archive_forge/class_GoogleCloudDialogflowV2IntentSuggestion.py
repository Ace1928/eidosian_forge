from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentSuggestion(_messages.Message):
    """Represents an intent suggestion.

  Fields:
    description: Human readable description for better understanding an intent
      like its scope, content, result etc. Maximum character limit: 140
      characters.
    displayName: The display name of the intent.
    intentV2: The unique identifier of this intent. Format:
      `projects//locations//agent/intents/`.
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    intentV2 = _messages.StringField(3)