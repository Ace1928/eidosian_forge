from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VideoStream(_messages.Message):
    """Video stream resource.

  Fields:
    h264: H264 codec settings.
    h265: H265 codec settings.
    vp9: VP9 codec settings.
  """
    h264 = _messages.MessageField('H264CodecSettings', 1)
    h265 = _messages.MessageField('H265CodecSettings', 2)
    vp9 = _messages.MessageField('Vp9CodecSettings', 3)