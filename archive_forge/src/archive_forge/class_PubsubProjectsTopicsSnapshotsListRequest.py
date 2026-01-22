from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsTopicsSnapshotsListRequest(_messages.Message):
    """A PubsubProjectsTopicsSnapshotsListRequest object.

  Fields:
    pageSize: Optional. Maximum number of snapshot names to return.
    pageToken: Optional. The value returned by the last
      `ListTopicSnapshotsResponse`; indicates that this is a continuation of a
      prior `ListTopicSnapshots` call, and that the system should return the
      next page of data.
    topic: Required. The name of the topic that snapshots are attached to.
      Format is `projects/{project}/topics/{topic}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    topic = _messages.StringField(3, required=True)