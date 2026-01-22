from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GraphicsStats(_messages.Message):
    """Graphics statistics for the App. The information is collected from 'adb
  shell dumpsys graphicsstats'. For more info see:
  https://developer.android.com/training/testing/performance.html Statistics
  will only be present for API 23+.

  Fields:
    buckets: Histogram of frame render times. There should be 154 buckets
      ranging from [5ms, 6ms) to [4950ms, infinity)
    highInputLatencyCount: Total "high input latency" events.
    jankyFrames: Total frames with slow render time. Should be <=
      total_frames.
    missedVsyncCount: Total "missed vsync" events.
    p50Millis: 50th percentile frame render time in milliseconds.
    p90Millis: 90th percentile frame render time in milliseconds.
    p95Millis: 95th percentile frame render time in milliseconds.
    p99Millis: 99th percentile frame render time in milliseconds.
    slowBitmapUploadCount: Total "slow bitmap upload" events.
    slowDrawCount: Total "slow draw" events.
    slowUiThreadCount: Total "slow UI thread" events.
    totalFrames: Total frames rendered by package.
  """
    buckets = _messages.MessageField('GraphicsStatsBucket', 1, repeated=True)
    highInputLatencyCount = _messages.IntegerField(2)
    jankyFrames = _messages.IntegerField(3)
    missedVsyncCount = _messages.IntegerField(4)
    p50Millis = _messages.IntegerField(5)
    p90Millis = _messages.IntegerField(6)
    p95Millis = _messages.IntegerField(7)
    p99Millis = _messages.IntegerField(8)
    slowBitmapUploadCount = _messages.IntegerField(9)
    slowDrawCount = _messages.IntegerField(10)
    slowUiThreadCount = _messages.IntegerField(11)
    totalFrames = _messages.IntegerField(12)