from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkerThreadScalingReportResponse(_messages.Message):
    """Contains the thread scaling recommendation for a worker from the
  backend.

  Fields:
    recommendedThreadCount: Recommended number of threads for a worker.
  """
    recommendedThreadCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)