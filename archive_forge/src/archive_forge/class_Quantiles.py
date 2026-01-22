from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Quantiles(_messages.Message):
    """Quantile metrics data related to Tasks. Units can be seconds, bytes,
  milliseconds, etc depending on the message type.

  Fields:
    count: A string attribute.
    maximum: A string attribute.
    minimum: A string attribute.
    percentile25: A string attribute.
    percentile50: A string attribute.
    percentile75: A string attribute.
    sum: A string attribute.
  """
    count = _messages.IntegerField(1)
    maximum = _messages.IntegerField(2)
    minimum = _messages.IntegerField(3)
    percentile25 = _messages.IntegerField(4)
    percentile50 = _messages.IntegerField(5)
    percentile75 = _messages.IntegerField(6)
    sum = _messages.IntegerField(7)