from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RankingMetrics(_messages.Message):
    """Evaluation metrics used by weighted-ALS models specified by
  feedback_type=implicit.

  Fields:
    averageRank: Determines the goodness of a ranking by computing the
      percentile rank from the predicted confidence and dividing it by the
      original rank.
    meanAveragePrecision: Calculates a precision per user for all the items by
      ranking them and then averages all the precisions across all the users.
    meanSquaredError: Similar to the mean squared error computed in regression
      and explicit recommendation models except instead of computing the
      rating directly, the output from evaluate is computed against a
      preference which is 1 or 0 depending on if the rating exists or not.
    normalizedDiscountedCumulativeGain: A metric to determine the goodness of
      a ranking calculated from the predicted confidence by comparing it to an
      ideal rank measured by the original ratings.
  """
    averageRank = _messages.FloatField(1)
    meanAveragePrecision = _messages.FloatField(2)
    meanSquaredError = _messages.FloatField(3)
    normalizedDiscountedCumulativeGain = _messages.FloatField(4)