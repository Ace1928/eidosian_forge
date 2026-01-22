from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsJobsSnapshotRequest(_messages.Message):
    """A DataflowProjectsJobsSnapshotRequest object.

  Fields:
    jobId: The job to be snapshotted.
    projectId: The project which owns the job to be snapshotted.
    snapshotJobRequest: A SnapshotJobRequest resource to be passed as the
      request body.
  """
    jobId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    snapshotJobRequest = _messages.MessageField('SnapshotJobRequest', 3)