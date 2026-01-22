from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RagFile(_messages.Message):
    """A RagFile contains user data for chunking, embedding and indexing.

  Enums:
    RagFileTypeValueValuesEnum: Output only. The type of the RagFile.

  Fields:
    createTime: Output only. Timestamp when this RagFile was created.
    description: Optional. The description of the RagFile.
    directUploadSource: Output only. The RagFile is encapsulated and uploaded
      in the UploadRagFile request.
    displayName: Required. The display name of the RagFile. The name can be up
      to 128 characters long and can consist of any UTF-8 characters.
    gcsSource: Output only. Google Cloud Storage location of the RagFile. It
      does not support wildcards in the GCS uri for now.
    googleDriveSource: Output only. Google Drive location. Supports importing
      individual files as well as Google Drive folders.
    name: Output only. The resource name of the RagFile.
    ragFileType: Output only. The type of the RagFile.
    sizeBytes: Output only. The size of the RagFile in bytes.
    updateTime: Output only. Timestamp when this RagFile was last updated.
  """

    class RagFileTypeValueValuesEnum(_messages.Enum):
        """Output only. The type of the RagFile.

    Values:
      RAG_FILE_TYPE_UNSPECIFIED: RagFile type is unspecified.
      RAG_FILE_TYPE_TXT: RagFile type is TXT.
      RAG_FILE_TYPE_PDF: RagFile type is PDF.
    """
        RAG_FILE_TYPE_UNSPECIFIED = 0
        RAG_FILE_TYPE_TXT = 1
        RAG_FILE_TYPE_PDF = 2
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    directUploadSource = _messages.MessageField('GoogleCloudAiplatformV1beta1DirectUploadSource', 3)
    displayName = _messages.StringField(4)
    gcsSource = _messages.MessageField('GoogleCloudAiplatformV1beta1GcsSource', 5)
    googleDriveSource = _messages.MessageField('GoogleCloudAiplatformV1beta1GoogleDriveSource', 6)
    name = _messages.StringField(7)
    ragFileType = _messages.EnumField('RagFileTypeValueValuesEnum', 8)
    sizeBytes = _messages.IntegerField(9)
    updateTime = _messages.StringField(10)