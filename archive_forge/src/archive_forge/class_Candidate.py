from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Candidate(_messages.Message):
    """A response candidate generated from the model.

  Enums:
    FinishReasonValueValuesEnum: Optional. Output only. The reason why the
      model stopped generating tokens. If empty, the model has not stopped
      generating the tokens.

  Fields:
    citationMetadata: Output only. Citation information for model-generated
      candidate. This field may be populated with recitation information for
      any text included in the `content`. These are passages that are
      "recited" from copyrighted material in the foundational LLM's training
      data.
    content: Output only. Generated content returned from the model.
    finishReason: Optional. Output only. The reason why the model stopped
      generating tokens. If empty, the model has not stopped generating the
      tokens.
    index: Output only. Index of the candidate in the list of candidates.
    safetyRatings: List of ratings for the safety of a response candidate.
      There is at most one rating per category.
    tokenCount: Output only. Token count for this candidate.
  """

    class FinishReasonValueValuesEnum(_messages.Enum):
        """Optional. Output only. The reason why the model stopped generating
    tokens. If empty, the model has not stopped generating the tokens.

    Values:
      FINISH_REASON_UNSPECIFIED: Default value. This value is unused.
      STOP: Natural stop point of the model or provided stop sequence.
      MAX_TOKENS: The maximum number of tokens as specified in the request was
        reached.
      SAFETY: The candidate content was flagged for safety reasons.
      RECITATION: The candidate content was flagged for recitation reasons.
      OTHER: Unknown reason.
    """
        FINISH_REASON_UNSPECIFIED = 0
        STOP = 1
        MAX_TOKENS = 2
        SAFETY = 3
        RECITATION = 4
        OTHER = 5
    citationMetadata = _messages.MessageField('CitationMetadata', 1)
    content = _messages.MessageField('Content', 2)
    finishReason = _messages.EnumField('FinishReasonValueValuesEnum', 3)
    index = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    safetyRatings = _messages.MessageField('SafetyRating', 5, repeated=True)
    tokenCount = _messages.IntegerField(6, variant=_messages.Variant.INT32)