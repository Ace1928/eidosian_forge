from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShuffleWriteQuantileMetrics(_messages.Message):
    """A ShuffleWriteQuantileMetrics object.

  Fields:
    writeBytes: A Quantiles attribute.
    writeRecords: A Quantiles attribute.
    writeTimeNanos: A Quantiles attribute.
  """
    writeBytes = _messages.MessageField('Quantiles', 1)
    writeRecords = _messages.MessageField('Quantiles', 2)
    writeTimeNanos = _messages.MessageField('Quantiles', 3)