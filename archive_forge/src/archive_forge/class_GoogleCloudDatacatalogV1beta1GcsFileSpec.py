from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1GcsFileSpec(_messages.Message):
    """Specifications of a single file in Cloud Storage.

  Fields:
    filePath: Required. The full file path. Example:
      `gs://bucket_name/a/b.txt`.
    gcsTimestamps: Output only. Timestamps about the Cloud Storage file.
    sizeBytes: Output only. The size of the file, in bytes.
  """
    filePath = _messages.StringField(1)
    gcsTimestamps = _messages.MessageField('GoogleCloudDatacatalogV1beta1SystemTimestamps', 2)
    sizeBytes = _messages.IntegerField(3)