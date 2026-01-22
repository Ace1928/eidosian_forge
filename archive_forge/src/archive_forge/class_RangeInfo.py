from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RangeInfo(_messages.Message):
    """RangeInfo contains the range name and the range utilization by this
  cluster.

  Fields:
    rangeName: Output only. [Output only] Name of a range.
    utilization: Output only. [Output only] The utilization of the range.
  """
    rangeName = _messages.StringField(1)
    utilization = _messages.FloatField(2)