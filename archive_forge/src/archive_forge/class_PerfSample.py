from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerfSample(_messages.Message):
    """Resource representing a single performance measure or data point

  Fields:
    sampleTime: Timestamp of collection.
    value: Value observed
  """
    sampleTime = _messages.MessageField('Timestamp', 1)
    value = _messages.FloatField(2)