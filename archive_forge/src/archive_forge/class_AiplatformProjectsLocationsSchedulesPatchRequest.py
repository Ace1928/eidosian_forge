from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsSchedulesPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsSchedulesPatchRequest object.

  Fields:
    googleCloudAiplatformV1Schedule: A GoogleCloudAiplatformV1Schedule
      resource to be passed as the request body.
    name: Immutable. The resource name of the Schedule.
    updateMask: Required. The update mask applies to the resource. See
      google.protobuf.FieldMask.
  """
    googleCloudAiplatformV1Schedule = _messages.MessageField('GoogleCloudAiplatformV1Schedule', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)