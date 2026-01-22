from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvaluationMetrics(_messages.Message):
    """Evaluation metrics of a model. These are either computed on all training
  data or just the eval data based on whether eval data was used during
  training. These are not present for imported models.

  Fields:
    arimaForecastingMetrics: Populated for ARIMA models.
    binaryClassificationMetrics: Populated for binary
      classification/classifier models.
    clusteringMetrics: Populated for clustering models.
    dimensionalityReductionMetrics: Evaluation metrics when the model is a
      dimensionality reduction model, which currently includes PCA.
    multiClassClassificationMetrics: Populated for multi-class
      classification/classifier models.
    rankingMetrics: Populated for implicit feedback type matrix factorization
      models.
    regressionMetrics: Populated for regression models and explicit feedback
      type matrix factorization models.
  """
    arimaForecastingMetrics = _messages.MessageField('ArimaForecastingMetrics', 1)
    binaryClassificationMetrics = _messages.MessageField('BinaryClassificationMetrics', 2)
    clusteringMetrics = _messages.MessageField('ClusteringMetrics', 3)
    dimensionalityReductionMetrics = _messages.MessageField('DimensionalityReductionMetrics', 4)
    multiClassClassificationMetrics = _messages.MessageField('MultiClassClassificationMetrics', 5)
    rankingMetrics = _messages.MessageField('RankingMetrics', 6)
    regressionMetrics = _messages.MessageField('RegressionMetrics', 7)