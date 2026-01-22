from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionProto(_messages.Message):
    """A rectangular region.

  Fields:
    heightPx: The height, in pixels. Always set.
    leftPx: The left side of the rectangle, in pixels. Always set.
    topPx: The top of the rectangle, in pixels. Always set.
    widthPx: The width, in pixels. Always set.
  """
    heightPx = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    leftPx = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    topPx = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    widthPx = _messages.IntegerField(4, variant=_messages.Variant.INT32)