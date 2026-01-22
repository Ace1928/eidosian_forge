from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsSchedulesResumeRequest(_messages.Message):
    """A AiplatformProjectsLocationsSchedulesResumeRequest object.

  Fields:
    googleCloudAiplatformV1ResumeScheduleRequest: A
      GoogleCloudAiplatformV1ResumeScheduleRequest resource to be passed as
      the request body.
    name: Required. The name of the Schedule resource to be resumed. Format:
      `projects/{project}/locations/{location}/schedules/{schedule}`
  """
    googleCloudAiplatformV1ResumeScheduleRequest = _messages.MessageField('GoogleCloudAiplatformV1ResumeScheduleRequest', 1)
    name = _messages.StringField(2, required=True)