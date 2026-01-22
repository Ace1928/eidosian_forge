from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2AutomatedAgentReply(_messages.Message):
    """Represents a response from an automated agent.

  Enums:
    AutomatedAgentReplyTypeValueValuesEnum: AutomatedAgentReply type.

  Fields:
    allowCancellation: Indicates whether the partial automated agent reply is
      interruptible when a later reply message arrives. e.g. if the agent
      specified some music as partial response, it can be cancelled.
    automatedAgentReplyType: AutomatedAgentReply type.
    cxCurrentPage: The unique identifier of the current Dialogflow CX
      conversation page. Format: `projects//locations//agents//flows//pages/`.
    detectIntentResponse: Response of the Dialogflow Sessions.DetectIntent
      call.
  """

    class AutomatedAgentReplyTypeValueValuesEnum(_messages.Enum):
        """AutomatedAgentReply type.

    Values:
      AUTOMATED_AGENT_REPLY_TYPE_UNSPECIFIED: Not specified. This should never
        happen.
      PARTIAL: Partial reply. e.g. Aggregated responses in a `Fulfillment`
        that enables `return_partial_response` can be returned as partial
        reply. WARNING: partial reply is not eligible for barge-in.
      FINAL: Final reply.
    """
        AUTOMATED_AGENT_REPLY_TYPE_UNSPECIFIED = 0
        PARTIAL = 1
        FINAL = 2
    allowCancellation = _messages.BooleanField(1)
    automatedAgentReplyType = _messages.EnumField('AutomatedAgentReplyTypeValueValuesEnum', 2)
    cxCurrentPage = _messages.StringField(3)
    detectIntentResponse = _messages.MessageField('GoogleCloudDialogflowV2DetectIntentResponse', 4)