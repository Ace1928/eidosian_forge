from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Candidate(_messages.Message):
    """A response candidate generated from the model.

  Enums:
    FinishReasonValueValuesEnum: Output only. The reason why the model stopped
      generating tokens. If empty, the model has not stopped generating the
      tokens.

  Fields:
    citationMetadata: Output only. Source attribution of the generated
      content.
    content: Output only. Content parts of the candidate.
    finishMessage: Output only. Describes the reason the mode stopped
      generating tokens in more detail. This is only filled when
      `finish_reason` is set.
    finishReason: Output only. The reason why the model stopped generating
      tokens. If empty, the model has not stopped generating the tokens.
    groundingMetadata: Output only. Metadata specifies sources used to ground
      generated content.
    index: Output only. Index of the candidate.
    safetyRatings: Output only. List of ratings for the safety of a response
      candidate. There is at most one rating per category.
  """

    class FinishReasonValueValuesEnum(_messages.Enum):
        """Output only. The reason why the model stopped generating tokens. If
    empty, the model has not stopped generating the tokens.

    Values:
      FINISH_REASON_UNSPECIFIED: The finish reason is unspecified.
      STOP: Natural stop point of the model or provided stop sequence.
      MAX_TOKENS: The maximum number of tokens as specified in the request was
        reached.
      SAFETY: The token generation was stopped as the response was flagged for
        safety reasons. NOTE: When streaming the Candidate.content will be
        empty if content filters blocked the output.
      RECITATION: The token generation was stopped as the response was flagged
        for unauthorized citations.
      OTHER: All other reasons that stopped the token generation
      BLOCKLIST: The token generation was stopped as the response was flagged
        for the terms which are included from the terminology blocklist.
      PROHIBITED_CONTENT: The token generation was stopped as the response was
        flagged for the prohibited contents.
      SPII: The token generation was stopped as the response was flagged for
        Sensitive Personally Identifiable Information (SPII) contents.
    """
        FINISH_REASON_UNSPECIFIED = 0
        STOP = 1
        MAX_TOKENS = 2
        SAFETY = 3
        RECITATION = 4
        OTHER = 5
        BLOCKLIST = 6
        PROHIBITED_CONTENT = 7
        SPII = 8
    citationMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1CitationMetadata', 1)
    content = _messages.MessageField('GoogleCloudAiplatformV1beta1Content', 2)
    finishMessage = _messages.StringField(3)
    finishReason = _messages.EnumField('FinishReasonValueValuesEnum', 4)
    groundingMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1GroundingMetadata', 5)
    index = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    safetyRatings = _messages.MessageField('GoogleCloudAiplatformV1beta1SafetyRating', 7, repeated=True)