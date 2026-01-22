from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ConversationTurn(_messages.Message):
    """One interaction between a human and virtual agent. The human provides
  some input and the virtual agent provides a response.

  Fields:
    userInput: The user input.
    virtualAgentOutput: The virtual agent output.
  """
    userInput = _messages.MessageField('GoogleCloudDialogflowCxV3ConversationTurnUserInput', 1)
    virtualAgentOutput = _messages.MessageField('GoogleCloudDialogflowCxV3ConversationTurnVirtualAgentOutput', 2)