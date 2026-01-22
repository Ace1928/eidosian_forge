from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnimationEnd(_messages.Message):
    """End previous overlay animation from the video. Without `AnimationEnd`,
  the overlay object will keep the state of previous animation until the end
  of the video.

  Fields:
    startTimeOffset: The time to end overlay object, in seconds. Default: 0
  """
    startTimeOffset = _messages.StringField(1)