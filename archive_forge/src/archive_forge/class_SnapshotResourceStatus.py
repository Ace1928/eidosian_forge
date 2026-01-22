from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SnapshotResourceStatus(_messages.Message):
    """A SnapshotResourceStatus object.

  Fields:
    scheduledDeletionTime: [Output only] Scheduled deletion time of the
      snapshot. The snapshot will be deleted by the at any point within one
      hour after the deletion time.
  """
    scheduledDeletionTime = _messages.StringField(1)