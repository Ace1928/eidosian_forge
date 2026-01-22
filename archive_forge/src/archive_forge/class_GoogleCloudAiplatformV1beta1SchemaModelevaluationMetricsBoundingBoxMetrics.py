from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsBoundingBoxMetrics(_messages.Message):
    """Bounding box matching model metrics for a single intersection-over-union
  threshold and multiple label match confidence thresholds.

  Fields:
    confidenceMetrics: Metrics for each label-match confidence_threshold from
      0.05,0.10,...,0.95,0.96,0.97,0.98,0.99. Precision-recall curve is
      derived from them.
    iouThreshold: The intersection-over-union threshold value used to compute
      this metrics entry.
    meanAveragePrecision: The mean average precision, most often close to
      `auPrc`.
  """
    confidenceMetrics = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsBoundingBoxMetricsConfidenceMetrics', 1, repeated=True)
    iouThreshold = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    meanAveragePrecision = _messages.FloatField(3, variant=_messages.Variant.FLOAT)