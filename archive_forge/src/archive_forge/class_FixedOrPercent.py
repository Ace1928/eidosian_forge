from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class FixedOrPercent(_messages.Message):
    """Message encapsulating a value that can be either absolute ("fixed") or
  relative ("percent") to a value.

  Fields:
    fixed: Specifies a fixed value.
    percent: Specifies the relative value defined as a percentage, which will
      be multiplied by a reference value.
  """
    fixed = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    percent = _messages.IntegerField(2, variant=_messages.Variant.INT32)