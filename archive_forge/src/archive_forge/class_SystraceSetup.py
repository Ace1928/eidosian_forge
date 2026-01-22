from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SystraceSetup(_messages.Message):
    """A SystraceSetup object.

  Fields:
    durationSeconds: Systrace duration in seconds. Should be between 1 and 30
      seconds. 0 disables systrace.
  """
    durationSeconds = _messages.IntegerField(1, variant=_messages.Variant.INT32)