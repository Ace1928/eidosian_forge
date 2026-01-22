from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KnowledgeTypesValueListEntryValuesEnum(_messages.Enum):
    """KnowledgeTypesValueListEntryValuesEnum enum type.

    Values:
      KNOWLEDGE_TYPE_UNSPECIFIED: The type is unspecified or arbitrary.
      FAQ: The document content contains question and answer pairs as either
        HTML or CSV. Typical FAQ HTML formats are parsed accurately, but
        unusual formats may fail to be parsed. CSV must have questions in the
        first column and answers in the second, with no header. Because of
        this explicit format, they are always parsed accurately.
      EXTRACTIVE_QA: Documents for which unstructured text is extracted and
        used for question answering.
      ARTICLE_SUGGESTION: The entire document content as a whole can be used
        for query results. Only for Contact Center Solutions on Dialogflow.
      AGENT_FACING_SMART_REPLY: The document contains agent-facing Smart Reply
        entries.
    """
    KNOWLEDGE_TYPE_UNSPECIFIED = 0
    FAQ = 1
    EXTRACTIVE_QA = 2
    ARTICLE_SUGGESTION = 3
    AGENT_FACING_SMART_REPLY = 4