from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1EvaluationMultiConfidenceMetrics(_messages.Message):
    """Metrics across multiple confidence levels.

  Enums:
    MetricsTypeValueValuesEnum: The metrics type for the label.

  Fields:
    auprc: The calculated area under the precision recall curve (AUPRC),
      computed by integrating over all confidence thresholds.
    auprcExact: The AUPRC for metrics with fuzzy matching disabled, i.e.,
      exact matching only.
    confidenceLevelMetrics: Metrics across confidence levels with fuzzy
      matching enabled.
    confidenceLevelMetricsExact: Metrics across confidence levels with only
      exact matching.
    estimatedCalibrationError: The Estimated Calibration Error (ECE) of the
      confidence of the predicted entities.
    estimatedCalibrationErrorExact: The ECE for the predicted entities with
      fuzzy matching disabled, i.e., exact matching only.
    metricsType: The metrics type for the label.
  """

    class MetricsTypeValueValuesEnum(_messages.Enum):
        """The metrics type for the label.

    Values:
      METRICS_TYPE_UNSPECIFIED: The metrics type is unspecified. By default,
        metrics without a particular specification are for leaf entity types
        (i.e., top-level entity types without child types, or child types
        which are not parent types themselves).
      AGGREGATE: Indicates whether metrics for this particular label type
        represent an aggregate of metrics for other types instead of being
        based on actual TP/FP/FN values for the label type. Metrics for parent
        (i.e., non-leaf) entity types are an aggregate of metrics for their
        children.
    """
        METRICS_TYPE_UNSPECIFIED = 0
        AGGREGATE = 1
    auprc = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    auprcExact = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    confidenceLevelMetrics = _messages.MessageField('GoogleCloudDocumentaiV1EvaluationConfidenceLevelMetrics', 3, repeated=True)
    confidenceLevelMetricsExact = _messages.MessageField('GoogleCloudDocumentaiV1EvaluationConfidenceLevelMetrics', 4, repeated=True)
    estimatedCalibrationError = _messages.FloatField(5, variant=_messages.Variant.FLOAT)
    estimatedCalibrationErrorExact = _messages.FloatField(6, variant=_messages.Variant.FLOAT)
    metricsType = _messages.EnumField('MetricsTypeValueValuesEnum', 7)