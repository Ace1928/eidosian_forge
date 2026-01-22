from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DistributionUpdate(_messages.Message):
    """A metric value representing a distribution.

  Fields:
    count: The count of the number of elements present in the distribution.
    histogram: (Optional) Histogram of value counts for the distribution.
    max: The maximum value present in the distribution.
    min: The minimum value present in the distribution.
    sum: Use an int64 since we'd prefer the added precision. If overflow is a
      common problem we can detect it and use an additional int64 or a double.
    sumOfSquares: Use a double since the sum of squares is likely to overflow
      int64.
  """
    count = _messages.MessageField('SplitInt64', 1)
    histogram = _messages.MessageField('Histogram', 2)
    max = _messages.MessageField('SplitInt64', 3)
    min = _messages.MessageField('SplitInt64', 4)
    sum = _messages.MessageField('SplitInt64', 5)
    sumOfSquares = _messages.FloatField(6)