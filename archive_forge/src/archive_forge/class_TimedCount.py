from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TimedCount(_messages.Message):
    """The number of errors in a given time period. All numbers are approximate
  since the error events are sampled before counting them.

  Fields:
    count: Approximate number of occurrences in the given time period.
    endTime: End of the time period to which `count` refers (excluded).
    startTime: Start of the time period to which `count` refers (included).
  """
    count = _messages.IntegerField(1)
    endTime = _messages.StringField(2)
    startTime = _messages.StringField(3)