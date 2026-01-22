from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppStartTime(_messages.Message):
    """A AppStartTime object.

  Fields:
    fullyDrawnTime: Optional. The time from app start to reaching the
      developer-reported "fully drawn" time. This is only stored if the app
      includes a call to Activity.reportFullyDrawn(). See
      https://developer.android.com/topic/performance/launch-time.html#time-
      full
    initialDisplayTime: The time from app start to the first displayed
      activity being drawn, as reported in Logcat. See
      https://developer.android.com/topic/performance/launch-time.html#time-
      initial
  """
    fullyDrawnTime = _messages.MessageField('Duration', 1)
    initialDisplayTime = _messages.MessageField('Duration', 2)