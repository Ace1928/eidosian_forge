from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsCopyRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsCopyRequest object.

  Fields:
    googleCloudAiplatformV1CopyModelRequest: A
      GoogleCloudAiplatformV1CopyModelRequest resource to be passed as the
      request body.
    parent: Required. The resource name of the Location into which to copy the
      Model. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1CopyModelRequest = _messages.MessageField('GoogleCloudAiplatformV1CopyModelRequest', 1)
    parent = _messages.StringField(2, required=True)