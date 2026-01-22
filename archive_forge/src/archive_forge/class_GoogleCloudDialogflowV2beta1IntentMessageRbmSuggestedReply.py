from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageRbmSuggestedReply(_messages.Message):
    """Rich Business Messaging (RBM) suggested reply that the user can click
  instead of typing in their own response.

  Fields:
    postbackData: Opaque payload that the Dialogflow receives in a user event
      when the user taps the suggested reply. This data will be also forwarded
      to webhook to allow performing custom business logic.
    text: Suggested reply text.
  """
    postbackData = _messages.StringField(1)
    text = _messages.StringField(2)