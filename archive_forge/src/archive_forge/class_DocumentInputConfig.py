from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentInputConfig(_messages.Message):
    """A document translation request input config.

  Fields:
    content: Document's content represented as a stream of bytes.
    gcsSource: Google Cloud Storage location. This must be a single file. For
      example: gs://example_bucket/example_file.pdf
    mimeType: Specifies the input document's mime_type. If not specified it
      will be determined using the file extension for gcs_source provided
      files. For a file provided through bytes content the mime_type must be
      provided. Currently supported mime types are: - application/pdf -
      application/vnd.openxmlformats-officedocument.wordprocessingml.document
      - application/vnd.openxmlformats-
      officedocument.presentationml.presentation -
      application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
  """
    content = _messages.BytesField(1)
    gcsSource = _messages.MessageField('GcsSource', 2)
    mimeType = _messages.StringField(3)