from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TriggererResource(_messages.Message):
    """Configuration for resources used by Airflow triggerers.

  Fields:
    count: Optional. The number of triggerers.
    cpu: Optional. CPU request and limit for a single Airflow triggerer
      replica.
    memoryGb: Optional. Memory (GB) request and limit for a single Airflow
      triggerer replica.
  """
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    cpu = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    memoryGb = _messages.FloatField(3, variant=_messages.Variant.FLOAT)