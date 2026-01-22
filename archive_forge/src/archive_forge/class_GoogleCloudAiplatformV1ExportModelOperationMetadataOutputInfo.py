from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExportModelOperationMetadataOutputInfo(_messages.Message):
    """Further describes the output of the ExportModel. Supplements
  ExportModelRequest.OutputConfig.

  Fields:
    artifactOutputUri: Output only. If the Model artifact is being exported to
      Google Cloud Storage this is the full path of the directory created,
      into which the Model files are being written to.
    imageOutputUri: Output only. If the Model image is being exported to
      Google Container Registry or Artifact Registry this is the full path of
      the image created.
  """
    artifactOutputUri = _messages.StringField(1)
    imageOutputUri = _messages.StringField(2)