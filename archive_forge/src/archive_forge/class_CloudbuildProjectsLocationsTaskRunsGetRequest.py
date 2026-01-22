from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsTaskRunsGetRequest(_messages.Message):
    """A CloudbuildProjectsLocationsTaskRunsGetRequest object.

  Fields:
    name: Required. The name of the TaskRun to retrieve. Format:
      projects/{project}/locations/{location}/taskRuns/{taskRun}
  """
    name = _messages.StringField(1, required=True)