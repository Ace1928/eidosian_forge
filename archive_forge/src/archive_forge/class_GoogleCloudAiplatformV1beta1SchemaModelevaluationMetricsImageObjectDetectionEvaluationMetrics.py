from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsImageObjectDetectionEvaluationMetrics(_messages.Message):
    """Metrics for image object detection evaluation results.

  Fields:
    boundingBoxMeanAveragePrecision: The single metric for bounding boxes
      evaluation: the `meanAveragePrecision` averaged over all
      `boundingBoxMetricsEntries`.
    boundingBoxMetrics: The bounding boxes match metrics for each
      intersection-over-union threshold 0.05,0.10,...,0.95,0.96,0.97,0.98,0.99
      and each label confidence threshold
      0.05,0.10,...,0.95,0.96,0.97,0.98,0.99 pair.
    evaluatedBoundingBoxCount: The total number of bounding boxes (i.e. summed
      over all images) the ground truth used to create this evaluation had.
  """
    boundingBoxMeanAveragePrecision = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    boundingBoxMetrics = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsBoundingBoxMetrics', 2, repeated=True)
    evaluatedBoundingBoxCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)