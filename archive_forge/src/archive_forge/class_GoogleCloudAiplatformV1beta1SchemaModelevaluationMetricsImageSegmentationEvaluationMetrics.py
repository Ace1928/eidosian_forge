from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsImageSegmentationEvaluationMetrics(_messages.Message):
    """Metrics for image segmentation evaluation results.

  Fields:
    confidenceMetricsEntries: Metrics for each confidenceThreshold in
      0.00,0.05,0.10,...,0.95,0.96,0.97,0.98,0.99 Precision-recall curve can
      be derived from it.
  """
    confidenceMetricsEntries = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsImageSegmentationEvaluationMetricsConfidenceMetricsEntry', 1, repeated=True)