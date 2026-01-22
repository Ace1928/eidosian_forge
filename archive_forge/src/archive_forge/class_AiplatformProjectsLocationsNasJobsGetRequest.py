from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNasJobsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsNasJobsGetRequest object.

  Fields:
    name: Required. The name of the NasJob resource. Format:
      `projects/{project}/locations/{location}/nasJobs/{nas_job}`
  """
    name = _messages.StringField(1, required=True)