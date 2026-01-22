from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsUploadRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsUploadRequest object.

  Fields:
    googleCloudAiplatformV1UploadModelRequest: A
      GoogleCloudAiplatformV1UploadModelRequest resource to be passed as the
      request body.
    parent: Required. The resource name of the Location into which to upload
      the Model. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1UploadModelRequest = _messages.MessageField('GoogleCloudAiplatformV1UploadModelRequest', 1)
    parent = _messages.StringField(2, required=True)