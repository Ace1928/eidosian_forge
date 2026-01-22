from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentFollowupIntentInfo(_messages.Message):
    """Represents a single followup intent in the chain.

  Fields:
    followupIntentName: The unique identifier of the followup intent. Format:
      `projects//agent/intents/`.
    parentFollowupIntentName: The unique identifier of the followup intent's
      parent. Format: `projects//agent/intents/`.
  """
    followupIntentName = _messages.StringField(1)
    parentFollowupIntentName = _messages.StringField(2)