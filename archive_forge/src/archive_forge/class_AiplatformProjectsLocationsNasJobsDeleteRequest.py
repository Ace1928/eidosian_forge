from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNasJobsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsNasJobsDeleteRequest object.

  Fields:
    name: Required. The name of the NasJob resource to be deleted. Format:
      `projects/{project}/locations/{location}/nasJobs/{nas_job}`
  """
    name = _messages.StringField(1, required=True)