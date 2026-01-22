from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkerResource(_messages.Message):
    """Configuration for resources used by Airflow workers.

  Fields:
    cpu: Optional. CPU request and limit for a single Airflow worker replica.
    maxCount: Optional. Maximum number of workers for autoscaling.
    memoryGb: Optional. Memory (GB) request and limit for a single Airflow
      worker replica.
    minCount: Optional. Minimum number of workers for autoscaling.
    storageGb: Optional. Storage (GB) request and limit for a single Airflow
      worker replica.
  """
    cpu = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    maxCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    memoryGb = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    minCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    storageGb = _messages.FloatField(5, variant=_messages.Variant.FLOAT)