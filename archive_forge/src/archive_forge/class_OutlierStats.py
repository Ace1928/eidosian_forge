from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutlierStats(_messages.Message):
    """Statistics for the underflow and overflow bucket.

  Fields:
    overflowCount: Number of values that are larger than the upper bound of
      the largest bucket.
    overflowMean: Mean of values in the overflow bucket.
    underflowCount: Number of values that are smaller than the lower bound of
      the smallest bucket.
    underflowMean: Mean of values in the undeflow bucket.
  """
    overflowCount = _messages.IntegerField(1)
    overflowMean = _messages.FloatField(2)
    underflowCount = _messages.IntegerField(3)
    underflowMean = _messages.FloatField(4)