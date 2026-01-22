from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeWindow(_messages.Message):
    """A time window specified by its `start_time` and `end_time`.

  Fields:
    endTime: End time of the time window (inclusive). If not specified, the
      current timestamp is used instead.
    startTime: Start time of the time window (exclusive).
  """
    endTime = _messages.StringField(1)
    startTime = _messages.StringField(2)