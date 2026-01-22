from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3BatchDeleteDocumentsMetadataIndividualBatchDeleteStatus(_messages.Message):
    """The status of each individual document in the batch delete process.

  Fields:
    documentId: The document id of the document.
    status: The status of deleting the document in storage.
  """
    documentId = _messages.MessageField('GoogleCloudDocumentaiV1beta3DocumentId', 1)
    status = _messages.MessageField('GoogleRpcStatus', 2)