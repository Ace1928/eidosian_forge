from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LimitStatus(_messages.Message):
    """UsageResourceAllowanceStatus detail about usage consumption.

  Fields:
    consumed: Output only. Accumulated consumption during
      `consumption_interval`.
    consumptionInterval: Output only. The consumption interval.
    limit: Output only. Limit value of a UsageResourceAllowance within its one
      duration.
  """
    consumed = _messages.FloatField(1)
    consumptionInterval = _messages.MessageField('Interval', 2)
    limit = _messages.FloatField(3)