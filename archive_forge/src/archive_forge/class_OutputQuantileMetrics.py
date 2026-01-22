from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutputQuantileMetrics(_messages.Message):
    """A OutputQuantileMetrics object.

  Fields:
    bytesWritten: A Quantiles attribute.
    recordsWritten: A Quantiles attribute.
  """
    bytesWritten = _messages.MessageField('Quantiles', 1)
    recordsWritten = _messages.MessageField('Quantiles', 2)