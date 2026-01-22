from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsPatchRequest object.

  Fields:
    googleCloudAiplatformV1Tensorboard: A GoogleCloudAiplatformV1Tensorboard
      resource to be passed as the request body.
    name: Output only. Name of the Tensorboard. Format:
      `projects/{project}/locations/{location}/tensorboards/{tensorboard}`
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the Tensorboard resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field is overwritten if it's in the mask. If the user does
      not provide a mask then all fields are overwritten if new values are
      specified.
  """
    googleCloudAiplatformV1Tensorboard = _messages.MessageField('GoogleCloudAiplatformV1Tensorboard', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)