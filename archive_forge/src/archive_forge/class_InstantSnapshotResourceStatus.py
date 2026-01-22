from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstantSnapshotResourceStatus(_messages.Message):
    """A InstantSnapshotResourceStatus object.

  Fields:
    storageSizeBytes: [Output Only] The storage size of this instant snapshot.
  """
    storageSizeBytes = _messages.IntegerField(1)