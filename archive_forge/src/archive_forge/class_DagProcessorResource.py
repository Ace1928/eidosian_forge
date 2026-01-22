from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DagProcessorResource(_messages.Message):
    """Configuration for resources used by Airflow DAG processors. This field
  is supported for Cloud Composer environments in versions
  composer-3.*.*-airflow-*.*.* and newer.

  Fields:
    count: Optional. The number of DAG processors. If not provided or set to
      0, a single DAG processor instance will be created.
    cpu: Optional. CPU request and limit for a single Airflow DAG processor
      replica.
    memoryGb: Optional. Memory (GB) request and limit for a single Airflow DAG
      processor replica.
    storageGb: Optional. Storage (GB) request and limit for a single Airflow
      DAG processor replica.
  """
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    cpu = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    memoryGb = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    storageGb = _messages.FloatField(4, variant=_messages.Variant.FLOAT)