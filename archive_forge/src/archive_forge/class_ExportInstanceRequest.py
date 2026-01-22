from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportInstanceRequest(_messages.Message):
    """Request options for exporting data of an Instance.

  Fields:
    encryptionConfig: Required. Encryption configuration (CMEK). For CMEK
      enabled instances it should be same as looker CMEK.
    gcsUri: The path to the folder in Google Cloud Storage where the export
      will be stored. The URI is in the form `gs://bucketName/folderName`.
  """
    encryptionConfig = _messages.MessageField('ExportEncryptionConfig', 1)
    gcsUri = _messages.StringField(2)