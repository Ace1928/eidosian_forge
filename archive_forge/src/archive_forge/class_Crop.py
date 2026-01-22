from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Crop(_messages.Message):
    """Video cropping configuration for the input video. The cropped input
  video is scaled to match the output resolution.

  Fields:
    bottomPixels: The number of pixels to crop from the bottom. The default is
      0.
    leftPixels: The number of pixels to crop from the left. The default is 0.
    rightPixels: The number of pixels to crop from the right. The default is
      0.
    topPixels: The number of pixels to crop from the top. The default is 0.
  """
    bottomPixels = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    leftPixels = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    rightPixels = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    topPixels = _messages.IntegerField(4, variant=_messages.Variant.INT32)