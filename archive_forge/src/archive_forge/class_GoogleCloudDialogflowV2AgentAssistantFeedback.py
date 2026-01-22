from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2AgentAssistantFeedback(_messages.Message):
    """Detail feedback of Agent Assist result.

  Enums:
    AnswerRelevanceValueValuesEnum: Optional. Whether or not the suggested
      answer is relevant. For example: * Query: "Can I change my mailing
      address?" * Suggested document says: "Items must be returned/exchanged
      within 60 days of the purchase date." * answer_relevance:
      AnswerRelevance.IRRELEVANT
    DocumentCorrectnessValueValuesEnum: Optional. Whether or not the
      information in the document is correct. For example: * Query: "Can I
      return the package in 2 days once received?" * Suggested document says:
      "Items must be returned/exchanged within 60 days of the purchase date."
      * Ground truth: "No return or exchange is allowed." *
      [document_correctness]: INCORRECT
    DocumentEfficiencyValueValuesEnum: Optional. Whether or not the suggested
      document is efficient. For example, if the document is poorly written,
      hard to understand, hard to use or too long to find useful information,
      document_efficiency is DocumentEfficiency.INEFFICIENT.

  Fields:
    answerRelevance: Optional. Whether or not the suggested answer is
      relevant. For example: * Query: "Can I change my mailing address?" *
      Suggested document says: "Items must be returned/exchanged within 60
      days of the purchase date." * answer_relevance:
      AnswerRelevance.IRRELEVANT
    documentCorrectness: Optional. Whether or not the information in the
      document is correct. For example: * Query: "Can I return the package in
      2 days once received?" * Suggested document says: "Items must be
      returned/exchanged within 60 days of the purchase date." * Ground truth:
      "No return or exchange is allowed." * [document_correctness]: INCORRECT
    documentEfficiency: Optional. Whether or not the suggested document is
      efficient. For example, if the document is poorly written, hard to
      understand, hard to use or too long to find useful information,
      document_efficiency is DocumentEfficiency.INEFFICIENT.
    knowledgeSearchFeedback: Optional. Feedback for knowledge search.
    summarizationFeedback: Optional. Feedback for conversation summarization.
  """

    class AnswerRelevanceValueValuesEnum(_messages.Enum):
        """Optional. Whether or not the suggested answer is relevant. For
    example: * Query: "Can I change my mailing address?" * Suggested document
    says: "Items must be returned/exchanged within 60 days of the purchase
    date." * answer_relevance: AnswerRelevance.IRRELEVANT

    Values:
      ANSWER_RELEVANCE_UNSPECIFIED: Answer relevance unspecified.
      IRRELEVANT: Answer is irrelevant to query.
      RELEVANT: Answer is relevant to query.
    """
        ANSWER_RELEVANCE_UNSPECIFIED = 0
        IRRELEVANT = 1
        RELEVANT = 2

    class DocumentCorrectnessValueValuesEnum(_messages.Enum):
        """Optional. Whether or not the information in the document is correct.
    For example: * Query: "Can I return the package in 2 days once received?"
    * Suggested document says: "Items must be returned/exchanged within 60
    days of the purchase date." * Ground truth: "No return or exchange is
    allowed." * [document_correctness]: INCORRECT

    Values:
      DOCUMENT_CORRECTNESS_UNSPECIFIED: Document correctness unspecified.
      INCORRECT: Information in document is incorrect.
      CORRECT: Information in document is correct.
    """
        DOCUMENT_CORRECTNESS_UNSPECIFIED = 0
        INCORRECT = 1
        CORRECT = 2

    class DocumentEfficiencyValueValuesEnum(_messages.Enum):
        """Optional. Whether or not the suggested document is efficient. For
    example, if the document is poorly written, hard to understand, hard to
    use or too long to find useful information, document_efficiency is
    DocumentEfficiency.INEFFICIENT.

    Values:
      DOCUMENT_EFFICIENCY_UNSPECIFIED: Document efficiency unspecified.
      INEFFICIENT: Document is inefficient.
      EFFICIENT: Document is efficient.
    """
        DOCUMENT_EFFICIENCY_UNSPECIFIED = 0
        INEFFICIENT = 1
        EFFICIENT = 2
    answerRelevance = _messages.EnumField('AnswerRelevanceValueValuesEnum', 1)
    documentCorrectness = _messages.EnumField('DocumentCorrectnessValueValuesEnum', 2)
    documentEfficiency = _messages.EnumField('DocumentEfficiencyValueValuesEnum', 3)
    knowledgeSearchFeedback = _messages.MessageField('GoogleCloudDialogflowV2AgentAssistantFeedbackKnowledgeSearchFeedback', 4)
    summarizationFeedback = _messages.MessageField('GoogleCloudDialogflowV2AgentAssistantFeedbackSummarizationFeedback', 5)