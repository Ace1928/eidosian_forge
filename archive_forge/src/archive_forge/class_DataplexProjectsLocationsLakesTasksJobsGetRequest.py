from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesTasksJobsGetRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesTasksJobsGetRequest object.

  Fields:
    name: Required. The resource name of the job: projects/{project_number}/lo
      cations/{location_id}/lakes/{lake_id}/tasks/{task_id}/jobs/{job_id}.
  """
    name = _messages.StringField(1, required=True)