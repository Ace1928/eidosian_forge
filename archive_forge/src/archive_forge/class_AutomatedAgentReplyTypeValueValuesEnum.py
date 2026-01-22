from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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