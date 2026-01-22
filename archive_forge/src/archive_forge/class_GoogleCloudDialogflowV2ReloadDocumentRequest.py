from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ReloadDocumentRequest(_messages.Message):
    """Request message for Documents.ReloadDocument.

  Fields:
    contentUri: Optional. The path of gcs source file for reloading document
      content. For now, only gcs uri is supported. For documents stored in
      Google Cloud Storage, these URIs must have the form `gs:///`.
    importGcsCustomMetadata: Optional. Whether to import custom metadata from
      Google Cloud Storage. Only valid when the document source is Google
      Cloud Storage URI.
    smartMessagingPartialUpdate: Optional. When enabled, the reload request is
      to apply partial update to the smart messaging allowlist.
  """
    contentUri = _messages.StringField(1)
    importGcsCustomMetadata = _messages.BooleanField(2)
    smartMessagingPartialUpdate = _messages.BooleanField(3)