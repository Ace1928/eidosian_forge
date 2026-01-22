from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LatencyDistribution(_messages.Message):
    """Describes measured latency distribution.

  Fields:
    latencyPercentiles: Representative latency percentiles.
  """
    latencyPercentiles = _messages.MessageField('LatencyPercentile', 1, repeated=True)