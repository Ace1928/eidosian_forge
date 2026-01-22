from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsSchedulesCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsSchedulesCreateRequest object.

  Fields:
    googleCloudAiplatformV1Schedule: A GoogleCloudAiplatformV1Schedule
      resource to be passed as the request body.
    parent: Required. The resource name of the Location to create the Schedule
      in. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1Schedule = _messages.MessageField('GoogleCloudAiplatformV1Schedule', 1)
    parent = _messages.StringField(2, required=True)