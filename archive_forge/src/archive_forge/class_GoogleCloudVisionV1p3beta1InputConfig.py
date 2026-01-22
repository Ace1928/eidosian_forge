from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p3beta1InputConfig(_messages.Message):
    """The desired input location and metadata.

  Fields:
    content: File content, represented as a stream of bytes. Note: As with all
      `bytes` fields, protobuffers use a pure binary representation, whereas
      JSON representations use base64. Currently, this field only works for
      BatchAnnotateFiles requests. It does not work for
      AsyncBatchAnnotateFiles requests.
    gcsSource: The Google Cloud Storage location to read the input from.
    mimeType: The type of the file. Currently only "application/pdf",
      "image/tiff" and "image/gif" are supported. Wildcards are not supported.
  """
    content = _messages.BytesField(1)
    gcsSource = _messages.MessageField('GoogleCloudVisionV1p3beta1GcsSource', 2)
    mimeType = _messages.StringField(3)