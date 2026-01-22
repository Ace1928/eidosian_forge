from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeTimeCursorRequest(_messages.Message):
    """Compute the corresponding cursor for a publish or event time in a topic
  partition.

  Fields:
    partition: Required. The partition for which we should compute the cursor.
    target: Required. The target publish or event time. Specifying a future
      time will return an unset cursor.
  """
    partition = _messages.IntegerField(1)
    target = _messages.MessageField('TimeTarget', 2)