from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocMetricConfig(_messages.Message):
    """Dataproc metric config.

  Fields:
    metrics: Required. Metrics sources to enable.
  """
    metrics = _messages.MessageField('Metric', 1, repeated=True)