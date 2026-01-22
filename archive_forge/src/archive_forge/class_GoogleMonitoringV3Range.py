from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleMonitoringV3Range(_messages.Message):
    """Range of numerical values within min and max.

  Fields:
    max: Range maximum.
    min: Range minimum.
  """
    max = _messages.FloatField(1)
    min = _messages.FloatField(2)