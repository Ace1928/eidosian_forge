from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNasJobsCancelRequest(_messages.Message):
    """A AiplatformProjectsLocationsNasJobsCancelRequest object.

  Fields:
    googleCloudAiplatformV1CancelNasJobRequest: A
      GoogleCloudAiplatformV1CancelNasJobRequest resource to be passed as the
      request body.
    name: Required. The name of the NasJob to cancel. Format:
      `projects/{project}/locations/{location}/nasJobs/{nas_job}`
  """
    googleCloudAiplatformV1CancelNasJobRequest = _messages.MessageField('GoogleCloudAiplatformV1CancelNasJobRequest', 1)
    name = _messages.StringField(2, required=True)