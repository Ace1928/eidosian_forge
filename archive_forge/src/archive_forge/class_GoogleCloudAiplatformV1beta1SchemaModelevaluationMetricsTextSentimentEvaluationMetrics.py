from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsTextSentimentEvaluationMetrics(_messages.Message):
    """Model evaluation metrics for text sentiment problems.

  Fields:
    confusionMatrix: Confusion matrix of the evaluation. Only set for
      ModelEvaluations, not for ModelEvaluationSlices.
    f1Score: The harmonic mean of recall and precision.
    linearKappa: Linear weighted kappa. Only set for ModelEvaluations, not for
      ModelEvaluationSlices.
    meanAbsoluteError: Mean absolute error. Only set for ModelEvaluations, not
      for ModelEvaluationSlices.
    meanSquaredError: Mean squared error. Only set for ModelEvaluations, not
      for ModelEvaluationSlices.
    precision: Precision.
    quadraticKappa: Quadratic weighted kappa. Only set for ModelEvaluations,
      not for ModelEvaluationSlices.
    recall: Recall.
  """
    confusionMatrix = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsConfusionMatrix', 1)
    f1Score = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    linearKappa = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    meanAbsoluteError = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    meanSquaredError = _messages.FloatField(5, variant=_messages.Variant.FLOAT)
    precision = _messages.FloatField(6, variant=_messages.Variant.FLOAT)
    quadraticKappa = _messages.FloatField(7, variant=_messages.Variant.FLOAT)
    recall = _messages.FloatField(8, variant=_messages.Variant.FLOAT)