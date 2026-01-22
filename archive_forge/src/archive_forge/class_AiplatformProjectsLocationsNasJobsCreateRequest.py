from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNasJobsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsNasJobsCreateRequest object.

  Fields:
    googleCloudAiplatformV1NasJob: A GoogleCloudAiplatformV1NasJob resource to
      be passed as the request body.
    parent: Required. The resource name of the Location to create the NasJob
      in. Format: `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1NasJob = _messages.MessageField('GoogleCloudAiplatformV1NasJob', 1)
    parent = _messages.StringField(2, required=True)