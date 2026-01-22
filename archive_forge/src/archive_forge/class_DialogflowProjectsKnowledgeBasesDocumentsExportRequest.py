from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsKnowledgeBasesDocumentsExportRequest(_messages.Message):
    """A DialogflowProjectsKnowledgeBasesDocumentsExportRequest object.

  Fields:
    googleCloudDialogflowV2ExportDocumentRequest: A
      GoogleCloudDialogflowV2ExportDocumentRequest resource to be passed as
      the request body.
    name: Required. The name of the document to export. Format:
      `projects//locations//knowledgeBases//documents/`.
  """
    googleCloudDialogflowV2ExportDocumentRequest = _messages.MessageField('GoogleCloudDialogflowV2ExportDocumentRequest', 1)
    name = _messages.StringField(2, required=True)