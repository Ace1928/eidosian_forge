from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2InputConfig(_messages.Message):
    """The desired input location and metadata.

  Fields:
    contents: Content in bytes, represented as a stream of bytes. Note: As
      with all `bytes` fields, proto buffer messages use a pure binary
      representation, whereas JSON representations use base64. This field only
      works for synchronous ProcessDocument method.
    gcsSource: The Google Cloud Storage location to read the input from. This
      must be a single file.
    mimeType: Required. Mimetype of the input. Current supported mimetypes are
      application/pdf, image/tiff, and image/gif. In addition,
      application/json type is supported for requests with
      ProcessDocumentRequest.automl_params field set. The JSON file needs to
      be in Document format.
  """
    contents = _messages.BytesField(1)
    gcsSource = _messages.MessageField('GoogleCloudDocumentaiV1beta2GcsSource', 2)
    mimeType = _messages.StringField(3)