from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FileData(_messages.Message):
    """URI based data.

  Fields:
    fileUri: Required. URI.
    mimeType: Required. The IANA standard MIME type of the source data.
  """
    fileUri = _messages.StringField(1)
    mimeType = _messages.StringField(2)