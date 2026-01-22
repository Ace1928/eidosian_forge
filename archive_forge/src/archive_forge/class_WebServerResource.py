from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebServerResource(_messages.Message):
    """Configuration for resources used by Airflow web server.

  Fields:
    count: Optional. The number of web server instances. If not provided or
      set to 0, a single web server instance will be created.
    cpu: Optional. CPU request and limit for Airflow web server.
    memoryGb: Optional. Memory (GB) request and limit for Airflow web server.
    storageGb: Optional. Storage (GB) request and limit for Airflow web
      server.
  """
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    cpu = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    memoryGb = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    storageGb = _messages.FloatField(4, variant=_messages.Variant.FLOAT)