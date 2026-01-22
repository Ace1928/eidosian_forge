from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StageOutputMetrics(_messages.Message):
    """Metrics about the output written by the stage.

  Fields:
    bytesWritten: A string attribute.
    recordsWritten: A string attribute.
  """
    bytesWritten = _messages.IntegerField(1)
    recordsWritten = _messages.IntegerField(2)