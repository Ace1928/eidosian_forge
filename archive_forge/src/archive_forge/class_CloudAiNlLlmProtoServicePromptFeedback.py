from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServicePromptFeedback(_messages.Message):
    """Content filter results for a prompt sent in the request.

  Enums:
    BlockReasonValueValuesEnum: Blocked reason.

  Fields:
    blockReason: Blocked reason.
    blockReasonMessage: A readable block reason message.
    safetyRatings: Safety ratings.
  """

    class BlockReasonValueValuesEnum(_messages.Enum):
        """Blocked reason.

    Values:
      BLOCKED_REASON_UNSPECIFIED: Unspecified blocked reason.
      SAFETY: Candidates blocked due to safety.
      OTHER: Candidates blocked due to other reason.
      BLOCKLIST: Candidates blocked due to the terms which are included from
        the terminology blocklist.
      PROHIBITED_CONTENT: Candidates blocked due to prohibited content.
    """
        BLOCKED_REASON_UNSPECIFIED = 0
        SAFETY = 1
        OTHER = 2
        BLOCKLIST = 3
        PROHIBITED_CONTENT = 4
    blockReason = _messages.EnumField('BlockReasonValueValuesEnum', 1)
    blockReasonMessage = _messages.StringField(2)
    safetyRatings = _messages.MessageField('CloudAiNlLlmProtoServiceSafetyRating', 3, repeated=True)