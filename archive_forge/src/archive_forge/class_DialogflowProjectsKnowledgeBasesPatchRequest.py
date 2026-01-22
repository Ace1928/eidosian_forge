from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsKnowledgeBasesPatchRequest(_messages.Message):
    """A DialogflowProjectsKnowledgeBasesPatchRequest object.

  Fields:
    googleCloudDialogflowV2KnowledgeBase: A
      GoogleCloudDialogflowV2KnowledgeBase resource to be passed as the
      request body.
    name: The knowledge base resource name. The name must be empty when
      creating a knowledge base. Format:
      `projects//locations//knowledgeBases/`.
    updateMask: Optional. Not specified means `update all`. Currently, only
      `display_name` can be updated, an InvalidArgument will be returned for
      attempting to update other fields.
  """
    googleCloudDialogflowV2KnowledgeBase = _messages.MessageField('GoogleCloudDialogflowV2KnowledgeBase', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)