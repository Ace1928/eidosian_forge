from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3ImportDocumentsMetadataIndividualImportStatus(_messages.Message):
    """The status of each individual document in the import process.

  Fields:
    inputGcsSource: The source Cloud Storage URI of the document.
    outputDocumentId: The document id of imported document if it was
      successful, otherwise empty.
    status: The status of the importing of the document.
  """
    inputGcsSource = _messages.StringField(1)
    outputDocumentId = _messages.MessageField('GoogleCloudDocumentaiV1beta3DocumentId', 2)
    status = _messages.MessageField('GoogleRpcStatus', 3)