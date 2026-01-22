from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1KnowledgeAnswersAnswer(_messages.Message):
    """An answer from Knowledge Connector.

  Enums:
    MatchConfidenceLevelValueValuesEnum: The system's confidence level that
      this knowledge answer is a good match for this conversational query.
      NOTE: The confidence level for a given `` pair may change without
      notice, as it depends on models that are constantly being improved.
      However, it will change less frequently than the confidence score below,
      and should be preferred for referencing the quality of an answer.

  Fields:
    answer: The piece of text from the `source` knowledge base document that
      answers this conversational query.
    faqQuestion: The corresponding FAQ question if the answer was extracted
      from a FAQ Document, empty otherwise.
    matchConfidence: The system's confidence score that this Knowledge answer
      is a good match for this conversational query. The range is from 0.0
      (completely uncertain) to 1.0 (completely certain). Note: The confidence
      score is likely to vary somewhat (possibly even for identical requests),
      as the underlying model is under constant improvement. It may be
      deprecated in the future. We recommend using `match_confidence_level`
      which should be generally more stable.
    matchConfidenceLevel: The system's confidence level that this knowledge
      answer is a good match for this conversational query. NOTE: The
      confidence level for a given `` pair may change without notice, as it
      depends on models that are constantly being improved. However, it will
      change less frequently than the confidence score below, and should be
      preferred for referencing the quality of an answer.
    source: Indicates which Knowledge Document this answer was extracted from.
      Format: `projects//knowledgeBases//documents/`.
  """

    class MatchConfidenceLevelValueValuesEnum(_messages.Enum):
        """The system's confidence level that this knowledge answer is a good
    match for this conversational query. NOTE: The confidence level for a
    given `` pair may change without notice, as it depends on models that are
    constantly being improved. However, it will change less frequently than
    the confidence score below, and should be preferred for referencing the
    quality of an answer.

    Values:
      MATCH_CONFIDENCE_LEVEL_UNSPECIFIED: Not specified.
      LOW: Indicates that the confidence is low.
      MEDIUM: Indicates our confidence is medium.
      HIGH: Indicates our confidence is high.
    """
        MATCH_CONFIDENCE_LEVEL_UNSPECIFIED = 0
        LOW = 1
        MEDIUM = 2
        HIGH = 3
    answer = _messages.StringField(1)
    faqQuestion = _messages.StringField(2)
    matchConfidence = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    matchConfidenceLevel = _messages.EnumField('MatchConfidenceLevelValueValuesEnum', 4)
    source = _messages.StringField(5)