from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsCreateRequest object.

  Fields:
    googleCloudAiplatformV1Tensorboard: A GoogleCloudAiplatformV1Tensorboard
      resource to be passed as the request body.
    parent: Required. The resource name of the Location to create the
      Tensorboard in. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1Tensorboard = _messages.MessageField('GoogleCloudAiplatformV1Tensorboard', 1)
    parent = _messages.StringField(2, required=True)