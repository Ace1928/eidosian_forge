from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegressionMetrics(_messages.Message):
    """Evaluation metrics for regression and explicit feedback type matrix
  factorization models.

  Fields:
    meanAbsoluteError: Mean absolute error.
    meanSquaredError: Mean squared error.
    meanSquaredLogError: Mean squared log error.
    medianAbsoluteError: Median absolute error.
    rSquared: R^2 score. This corresponds to r2_score in ML.EVALUATE.
  """
    meanAbsoluteError = _messages.FloatField(1)
    meanSquaredError = _messages.FloatField(2)
    meanSquaredLogError = _messages.FloatField(3)
    medianAbsoluteError = _messages.FloatField(4)
    rSquared = _messages.FloatField(5)