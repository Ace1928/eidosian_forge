from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentKnowledgeBasesCreateRequest(_messages.Message):
    """A DialogflowProjectsAgentKnowledgeBasesCreateRequest object.

  Fields:
    googleCloudDialogflowV2KnowledgeBase: A
      GoogleCloudDialogflowV2KnowledgeBase resource to be passed as the
      request body.
    parent: Required. The project to create a knowledge base for. Format:
      `projects//locations/`.
  """
    googleCloudDialogflowV2KnowledgeBase = _messages.MessageField('GoogleCloudDialogflowV2KnowledgeBase', 1)
    parent = _messages.StringField(2, required=True)