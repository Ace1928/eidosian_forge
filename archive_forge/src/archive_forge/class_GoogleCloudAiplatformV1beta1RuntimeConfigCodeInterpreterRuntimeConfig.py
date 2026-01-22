from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RuntimeConfigCodeInterpreterRuntimeConfig(_messages.Message):
    """A GoogleCloudAiplatformV1beta1RuntimeConfigCodeInterpreterRuntimeConfig
  object.

  Fields:
    fileInputGcsBucket: Optional. The GCS bucket for file input of this
      Extension. If specified, support input from the GCS bucket. Vertex
      Extension Custom Code Service Agent should be granted file reader to
      this bucket. If not specified, the extension will only accept file
      contents from request body and reject GCS file inputs.
    fileOutputGcsBucket: Optional. The GCS bucket for file output of this
      Extension. If specified, write all output files to the GCS bucket.
      Vertex Extension Custom Code Service Agent should be granted file writer
      to this bucket. If not specified, the file content will be output in
      response body.
  """
    fileInputGcsBucket = _messages.StringField(1)
    fileOutputGcsBucket = _messages.StringField(2)