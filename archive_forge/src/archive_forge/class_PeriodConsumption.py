from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PeriodConsumption(_messages.Message):
    """A PeriodConsumption object.

  Fields:
    consumed: Output only. Accumulated consumption during
      `consumption_interval`.
    consumptionInterval: Output only. The consumption interval.
  """
    consumed = _messages.FloatField(1)
    consumptionInterval = _messages.MessageField('Interval', 2)