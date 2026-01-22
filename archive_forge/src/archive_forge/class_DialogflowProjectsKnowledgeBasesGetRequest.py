from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsKnowledgeBasesGetRequest(_messages.Message):
    """A DialogflowProjectsKnowledgeBasesGetRequest object.

  Fields:
    name: Required. The name of the knowledge base to retrieve. Format
      `projects//locations//knowledgeBases/`.
  """
    name = _messages.StringField(1, required=True)