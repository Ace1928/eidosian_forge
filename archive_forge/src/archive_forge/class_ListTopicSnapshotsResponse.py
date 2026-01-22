from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListTopicSnapshotsResponse(_messages.Message):
    """Response for the `ListTopicSnapshots` method.

  Fields:
    nextPageToken: Optional. If not empty, indicates that there may be more
      snapshots that match the request; this value should be passed in a new
      `ListTopicSnapshotsRequest` to get more snapshots.
    snapshots: Optional. The names of the snapshots that match the request.
  """
    nextPageToken = _messages.StringField(1)
    snapshots = _messages.StringField(2, repeated=True)