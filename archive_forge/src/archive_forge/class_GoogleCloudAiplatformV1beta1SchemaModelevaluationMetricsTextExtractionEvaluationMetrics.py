from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsTextExtractionEvaluationMetrics(_messages.Message):
    """Metrics for text extraction evaluation results.

  Fields:
    confidenceMetrics: Metrics that have confidence thresholds. Precision-
      recall curve can be derived from them.
    confusionMatrix: Confusion matrix of the evaluation. Only set for Models
      where number of AnnotationSpecs is no more than 10. Only set for
      ModelEvaluations, not for ModelEvaluationSlices.
  """
    confidenceMetrics = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsTextExtractionEvaluationMetricsConfidenceMetrics', 1, repeated=True)
    confusionMatrix = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsConfusionMatrix', 2)