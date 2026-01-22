from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsKnowledgeBasesDocumentsImportRequest(_messages.Message):
    """A DialogflowProjectsLocationsKnowledgeBasesDocumentsImportRequest
  object.

  Fields:
    googleCloudDialogflowV2ImportDocumentsRequest: A
      GoogleCloudDialogflowV2ImportDocumentsRequest resource to be passed as
      the request body.
    parent: Required. The knowledge base to import documents into. Format:
      `projects//locations//knowledgeBases/`.
  """
    googleCloudDialogflowV2ImportDocumentsRequest = _messages.MessageField('GoogleCloudDialogflowV2ImportDocumentsRequest', 1)
    parent = _messages.StringField(2, required=True)