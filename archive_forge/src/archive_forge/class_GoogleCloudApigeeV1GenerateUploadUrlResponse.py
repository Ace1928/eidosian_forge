from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1GenerateUploadUrlResponse(_messages.Message):
    """Response for GenerateUploadUrl method.

  Fields:
    uploadUri: The Google Cloud Storage signed URL that can be used to upload
      a new Archive zip file.
  """
    uploadUri = _messages.StringField(1)