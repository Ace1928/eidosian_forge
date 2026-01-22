from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsBoundingBoxMetricsConfidenceMetrics(_messages.Message):
    """Metrics for a single confidence threshold.

  Fields:
    confidenceThreshold: The confidence threshold value used to compute the
      metrics.
    f1Score: The harmonic mean of recall and precision.
    precision: Precision under the given confidence threshold.
    recall: Recall under the given confidence threshold.
  """
    confidenceThreshold = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    f1Score = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    precision = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    recall = _messages.FloatField(4, variant=_messages.Variant.FLOAT)