from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StageInputMetrics(_messages.Message):
    """Metrics about the input read by the stage.

  Fields:
    bytesRead: A string attribute.
    recordsRead: A string attribute.
  """
    bytesRead = _messages.IntegerField(1)
    recordsRead = _messages.IntegerField(2)