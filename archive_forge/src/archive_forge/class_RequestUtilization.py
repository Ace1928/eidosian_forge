from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RequestUtilization(_messages.Message):
    """Target scaling by request utilization. Only applicable in the App Engine
  flexible environment.

  Fields:
    targetConcurrentRequests: Target number of concurrent requests.
    targetRequestCountPerSecond: Target requests per second.
  """
    targetConcurrentRequests = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    targetRequestCountPerSecond = _messages.IntegerField(2, variant=_messages.Variant.INT32)