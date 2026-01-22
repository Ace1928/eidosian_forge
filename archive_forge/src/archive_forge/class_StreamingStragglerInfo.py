from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingStragglerInfo(_messages.Message):
    """Information useful for streaming straggler identification and debugging.

  Fields:
    dataWatermarkLag: The event-time watermark lag at the time of the
      straggler detection.
    endTime: End time of this straggler.
    startTime: Start time of this straggler.
    systemWatermarkLag: The system watermark lag at the time of the straggler
      detection.
    workerName: Name of the worker where the straggler was detected.
  """
    dataWatermarkLag = _messages.StringField(1)
    endTime = _messages.StringField(2)
    startTime = _messages.StringField(3)
    systemWatermarkLag = _messages.StringField(4)
    workerName = _messages.StringField(5)