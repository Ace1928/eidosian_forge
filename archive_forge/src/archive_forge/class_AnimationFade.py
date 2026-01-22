from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnimationFade(_messages.Message):
    """Display overlay object with fade animation.

  Enums:
    FadeTypeValueValuesEnum: Required. Type of fade animation: `FADE_IN` or
      `FADE_OUT`.

  Fields:
    endTimeOffset: The time to end the fade animation, in seconds. Default:
      `start_time_offset` + 1s
    fadeType: Required. Type of fade animation: `FADE_IN` or `FADE_OUT`.
    startTimeOffset: The time to start the fade animation, in seconds.
      Default: 0
    xy: Normalized coordinates based on output video resolution. Valid values:
      `0.0`\\u2013`1.0`. `xy` is the upper-left coordinate of the overlay
      object. For example, use the x and y coordinates {0,0} to position the
      top-left corner of the overlay animation in the top-left corner of the
      output video.
  """

    class FadeTypeValueValuesEnum(_messages.Enum):
        """Required. Type of fade animation: `FADE_IN` or `FADE_OUT`.

    Values:
      FADE_TYPE_UNSPECIFIED: The fade type is not specified.
      FADE_IN: Fade the overlay object into view.
      FADE_OUT: Fade the overlay object out of view.
    """
        FADE_TYPE_UNSPECIFIED = 0
        FADE_IN = 1
        FADE_OUT = 2
    endTimeOffset = _messages.StringField(1)
    fadeType = _messages.EnumField('FadeTypeValueValuesEnum', 2)
    startTimeOffset = _messages.StringField(3)
    xy = _messages.MessageField('NormalizedCoordinate', 4)