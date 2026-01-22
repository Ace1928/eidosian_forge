from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PerformDiskShrinkContext(_messages.Message):
    """Perform disk shrink context.

  Fields:
    targetSizeGb: The target disk shrink size in GigaBytes.
  """
    targetSizeGb = _messages.IntegerField(1)