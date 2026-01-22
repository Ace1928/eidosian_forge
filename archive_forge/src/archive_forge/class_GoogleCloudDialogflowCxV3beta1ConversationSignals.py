from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1ConversationSignals(_messages.Message):
    """This message is used to hold all the Conversation Signals data, which
  will be converted to JSON and exported to BigQuery.

  Fields:
    turnSignals: Required. Turn signals for the current turn.
  """
    turnSignals = _messages.MessageField('GoogleCloudDialogflowCxV3beta1TurnSignals', 1)