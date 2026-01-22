from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsDeleteSnapshotsRequest(_messages.Message):
    """A DataflowProjectsDeleteSnapshotsRequest object.

  Fields:
    location: The location that contains this snapshot.
    projectId: The ID of the Cloud Platform project that the snapshot belongs
      to.
    snapshotId: The ID of the snapshot.
  """
    location = _messages.StringField(1)
    projectId = _messages.StringField(2, required=True)
    snapshotId = _messages.StringField(3)