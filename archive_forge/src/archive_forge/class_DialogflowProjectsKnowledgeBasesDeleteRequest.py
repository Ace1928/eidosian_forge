from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsKnowledgeBasesDeleteRequest(_messages.Message):
    """A DialogflowProjectsKnowledgeBasesDeleteRequest object.

  Fields:
    force: Optional. Force deletes the knowledge base. When set to true, any
      documents in the knowledge base are also deleted.
    name: Required. The name of the knowledge base to delete. Format:
      `projects//locations//knowledgeBases/`.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)