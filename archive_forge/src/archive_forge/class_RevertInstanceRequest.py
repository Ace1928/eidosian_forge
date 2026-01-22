from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RevertInstanceRequest(_messages.Message):
    """RevertInstanceRequest reverts the given instance's file share to the
  specified snapshot.

  Fields:
    targetSnapshotId: Required. The snapshot resource ID, in the format 'my-
      snapshot', where the specified ID is the {snapshot_id} of the fully
      qualified name like `projects/{project_id}/locations/{location_id}/insta
      nces/{instance_id}/snapshots/{snapshot_id}`
  """
    targetSnapshotId = _messages.StringField(1)