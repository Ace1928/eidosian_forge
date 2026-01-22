from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3AutoLabelDocumentsMetadataIndividualAutoLabelStatus(_messages.Message):
    """The status of individual documents in the auto-labeling process.

  Fields:
    documentId: The document id of the auto-labeled document. This will
      replace the gcs_uri.
    status: The status of the document auto-labeling.
  """
    documentId = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3DocumentId', 1)
    status = _messages.MessageField('GoogleRpcStatus', 2)