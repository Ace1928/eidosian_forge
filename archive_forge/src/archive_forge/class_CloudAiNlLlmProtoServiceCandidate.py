from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServiceCandidate(_messages.Message):
    """A CloudAiNlLlmProtoServiceCandidate object.

  Enums:
    FinishReasonValueValuesEnum: The reason why the model stopped generating
      tokens.

  Fields:
    citationMetadata: Source attribution of the generated content.
    content: Content of the candidate.
    finishMessage: A string that describes the filtering behavior in more
      detail. Only filled when reason is set.
    finishReason: The reason why the model stopped generating tokens.
    groundingMetadata: Grounding metadata. Combine with the facts list from
      response to generate grounding citations for this choice.
    index: Index of the candidate.
    safetyRatings: Safety ratings of the generated content.
  """

    class FinishReasonValueValuesEnum(_messages.Enum):
        """The reason why the model stopped generating tokens.

    Values:
      FINISH_REASON_UNSPECIFIED: The finish reason is unspecified.
      FINISH_REASON_STOP: Natural stop point of the model or provided stop
        sequence.
      FINISH_REASON_MAX_TOKENS: The maximum number of tokens as specified in
        the request was reached.
      FINISH_REASON_SAFETY: The token generation was stopped as the response
        was flagged for safety reasons. NOTE: When streaming the
        Candidate.content will be empty if content filters blocked the output.
      FINISH_REASON_RECITATION: The token generation was stopped as the
        response was flagged for unauthorized citations.
      FINISH_REASON_OTHER: All other reasons that stopped the token
        generation.
      FINISH_REASON_BLOCKLIST: The token generation was stopped as the
        response was flagged for the terms which are included from the
        terminology blocklist.
      FINISH_REASON_PROHIBITED_CONTENT: The token generation was stopped as
        the response was flagged for the prohibited contents.
      FINISH_REASON_SPII: The token generation was stopped as the response was
        flagged for Sensitive Personally Identifiable Information (SPII)
        contents.
    """
        FINISH_REASON_UNSPECIFIED = 0
        FINISH_REASON_STOP = 1
        FINISH_REASON_MAX_TOKENS = 2
        FINISH_REASON_SAFETY = 3
        FINISH_REASON_RECITATION = 4
        FINISH_REASON_OTHER = 5
        FINISH_REASON_BLOCKLIST = 6
        FINISH_REASON_PROHIBITED_CONTENT = 7
        FINISH_REASON_SPII = 8
    citationMetadata = _messages.MessageField('CloudAiNlLlmProtoServiceCitationMetadata', 1)
    content = _messages.MessageField('CloudAiNlLlmProtoServiceContent', 2)
    finishMessage = _messages.StringField(3)
    finishReason = _messages.EnumField('FinishReasonValueValuesEnum', 4)
    groundingMetadata = _messages.MessageField('LearningGenaiRootGroundingMetadata', 5)
    index = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    safetyRatings = _messages.MessageField('CloudAiNlLlmProtoServiceSafetyRating', 7, repeated=True)