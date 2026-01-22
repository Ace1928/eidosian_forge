from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionQueryConfigDocumentQuerySource(_messages.Message):
    """Document source settings. Supported features: SMART_REPLY,
  SMART_COMPOSE.

  Fields:
    documents: Required. Knowledge documents to query from. Format:
      `projects//locations//knowledgeBases//documents/`. Currently, at most 5
      documents are supported.
  """
    documents = _messages.StringField(1, repeated=True)