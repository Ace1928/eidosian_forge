from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSnapshotsCreateRequest(_messages.Message):
    """A PubsubProjectsSnapshotsCreateRequest object.

  Fields:
    createSnapshotRequest: A CreateSnapshotRequest resource to be passed as
      the request body.
    name: Required. User-provided name for this snapshot. If the name is not
      provided in the request, the server will assign a random name for this
      snapshot on the same project as the subscription. Note that for REST API
      requests, you must specify a name. See the [resource name
      rules](https://cloud.google.com/pubsub/docs/pubsub-
      basics#resource_names). Format is `projects/{project}/snapshots/{snap}`.
  """
    createSnapshotRequest = _messages.MessageField('CreateSnapshotRequest', 1)
    name = _messages.StringField(2, required=True)