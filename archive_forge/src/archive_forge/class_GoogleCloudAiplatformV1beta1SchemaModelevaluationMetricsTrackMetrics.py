from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsTrackMetrics(_messages.Message):
    """UNIMPLEMENTED. Track matching model metrics for a single track match
  threshold and multiple label match confidence thresholds.

  Fields:
    confidenceMetrics: Metrics for each label-match `confidenceThreshold` from
      0.05,0.10,...,0.95,0.96,0.97,0.98,0.99. Precision-recall curve is
      derived from them.
    iouThreshold: The intersection-over-union threshold value between bounding
      boxes across frames used to compute this metric entry.
    meanBoundingBoxIou: The mean bounding box iou over all confidence
      thresholds.
    meanMismatchRate: The mean mismatch rate over all confidence thresholds.
    meanTrackingAveragePrecision: The mean average precision over all
      confidence thresholds.
  """
    confidenceMetrics = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsTrackMetricsConfidenceMetrics', 1, repeated=True)
    iouThreshold = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    meanBoundingBoxIou = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    meanMismatchRate = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    meanTrackingAveragePrecision = _messages.FloatField(5, variant=_messages.Variant.FLOAT)