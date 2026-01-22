from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageLinkOutSuggestion(_messages.Message):
    """The suggestion chip message that allows the user to jump out to the app
  or website associated with this agent.

  Fields:
    destinationName: Required. The name of the app or site this chip is
      linking to.
    uri: Required. The URI of the app or site to open when the user taps the
      suggestion chip.
  """
    destinationName = _messages.StringField(1)
    uri = _messages.StringField(2)