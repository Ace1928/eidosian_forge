from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsJobsSnapshotsListRequest(_messages.Message):
    """A DataflowProjectsLocationsJobsSnapshotsListRequest object.

  Fields:
    jobId: If specified, list snapshots created from this job.
    location: The location to list snapshots in.
    projectId: The project ID to list snapshots for.
  """
    jobId = _messages.StringField(1, required=True)
    location = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)