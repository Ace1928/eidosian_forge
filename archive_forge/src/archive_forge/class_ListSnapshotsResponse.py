from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSnapshotsResponse(_messages.Message):
    """List of snapshots.

  Fields:
    snapshots: Returned snapshots.
  """
    snapshots = _messages.MessageField('Snapshot', 1, repeated=True)