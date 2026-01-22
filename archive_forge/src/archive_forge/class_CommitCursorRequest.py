from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommitCursorRequest(_messages.Message):
    """Request for CommitCursor.

  Fields:
    cursor: The new value for the committed cursor.
    partition: The partition for which to update the cursor. Partitions are
      zero indexed, so `partition` must be in the range [0,
      topic.num_partitions).
  """
    cursor = _messages.MessageField('Cursor', 1)
    partition = _messages.IntegerField(2)