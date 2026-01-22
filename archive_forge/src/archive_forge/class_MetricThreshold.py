from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricThreshold(_messages.Message):
    """A condition type that compares a collection of time series against a
  threshold.

  Enums:
    ComparisonValueValuesEnum: The comparison to apply between the time series
      (indicated by filter and aggregation) and the threshold (indicated by
      threshold_value). The comparison is applied on each time series, with
      the time series on the left-hand side and the threshold on the right-
      hand side.Only COMPARISON_LT and COMPARISON_GT are supported currently.
    EvaluationMissingDataValueValuesEnum: A condition control that determines
      how metric-threshold conditions are evaluated when data stops arriving.

  Fields:
    aggregations: Specifies the alignment of data points in individual time
      series as well as how to combine the retrieved time series together
      (such as when aggregating multiple streams on each resource to a single
      stream for each resource or when aggregating streams across all members
      of a group of resources). Multiple aggregations are applied in the order
      specified.This field is similar to the one in the ListTimeSeries request
      (https://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.timeSer
      ies/list). It is advisable to use the ListTimeSeries method when
      debugging this field.
    comparison: The comparison to apply between the time series (indicated by
      filter and aggregation) and the threshold (indicated by
      threshold_value). The comparison is applied on each time series, with
      the time series on the left-hand side and the threshold on the right-
      hand side.Only COMPARISON_LT and COMPARISON_GT are supported currently.
    denominatorAggregations: Specifies the alignment of data points in
      individual time series selected by denominatorFilter as well as how to
      combine the retrieved time series together (such as when aggregating
      multiple streams on each resource to a single stream for each resource
      or when aggregating streams across all members of a group of
      resources).When computing ratios, the aggregations and
      denominator_aggregations fields must use the same alignment period and
      produce time series that have the same periodicity and labels.
    denominatorFilter: A filter
      (https://cloud.google.com/monitoring/api/v3/filters) that identifies a
      time series that should be used as the denominator of a ratio that will
      be compared with the threshold. If a denominator_filter is specified,
      the time series specified by the filter field will be used as the
      numerator.The filter must specify the metric type and optionally may
      contain restrictions on resource type, resource labels, and metric
      labels. This field may not exceed 2048 Unicode characters in length.
    duration: The amount of time that a time series must violate the threshold
      to be considered failing. Currently, only values that are a multiple of
      a minute--e.g., 0, 60, 120, or 300 seconds--are supported. If an invalid
      value is given, an error will be returned. When choosing a duration, it
      is useful to keep in mind the frequency of the underlying time series
      data (which may also be affected by any alignments specified in the
      aggregations field); a good duration is long enough so that a single
      outlier does not generate spurious alerts, but short enough that
      unhealthy states are detected and alerted on quickly.
    evaluationMissingData: A condition control that determines how metric-
      threshold conditions are evaluated when data stops arriving.
    filter: Required. A filter
      (https://cloud.google.com/monitoring/api/v3/filters) that identifies
      which time series should be compared with the threshold.The filter is
      similar to the one that is specified in the ListTimeSeries request (http
      s://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.timeSeries/l
      ist) (that call is useful to verify the time series that will be
      retrieved / processed). The filter must specify the metric type and the
      resource type. Optionally, it can specify resource labels and metric
      labels. This field must not exceed 2048 Unicode characters in length.
    forecastOptions: When this field is present, the MetricThreshold condition
      forecasts whether the time series is predicted to violate the threshold
      within the forecast_horizon. When this field is not set, the
      MetricThreshold tests the current value of the timeseries against the
      threshold.
    thresholdValue: A value against which to compare the time series.
    trigger: The number/percent of time series for which the comparison must
      hold in order for the condition to trigger. If unspecified, then the
      condition will trigger if the comparison is true for any of the time
      series that have been identified by filter and aggregations, or by the
      ratio, if denominator_filter and denominator_aggregations are specified.
  """

    class ComparisonValueValuesEnum(_messages.Enum):
        """The comparison to apply between the time series (indicated by filter
    and aggregation) and the threshold (indicated by threshold_value). The
    comparison is applied on each time series, with the time series on the
    left-hand side and the threshold on the right-hand side.Only COMPARISON_LT
    and COMPARISON_GT are supported currently.

    Values:
      COMPARISON_UNSPECIFIED: No ordering relationship is specified.
      COMPARISON_GT: True if the left argument is greater than the right
        argument.
      COMPARISON_GE: True if the left argument is greater than or equal to the
        right argument.
      COMPARISON_LT: True if the left argument is less than the right
        argument.
      COMPARISON_LE: True if the left argument is less than or equal to the
        right argument.
      COMPARISON_EQ: True if the left argument is equal to the right argument.
      COMPARISON_NE: True if the left argument is not equal to the right
        argument.
    """
        COMPARISON_UNSPECIFIED = 0
        COMPARISON_GT = 1
        COMPARISON_GE = 2
        COMPARISON_LT = 3
        COMPARISON_LE = 4
        COMPARISON_EQ = 5
        COMPARISON_NE = 6

    class EvaluationMissingDataValueValuesEnum(_messages.Enum):
        """A condition control that determines how metric-threshold conditions
    are evaluated when data stops arriving.

    Values:
      EVALUATION_MISSING_DATA_UNSPECIFIED: An unspecified evaluation missing
        data option. Equivalent to EVALUATION_MISSING_DATA_NO_OP.
      EVALUATION_MISSING_DATA_INACTIVE: If there is no data to evaluate the
        condition, then evaluate the condition as false.
      EVALUATION_MISSING_DATA_ACTIVE: If there is no data to evaluate the
        condition, then evaluate the condition as true.
      EVALUATION_MISSING_DATA_NO_OP: Do not evaluate the condition to any
        value if there is no data.
    """
        EVALUATION_MISSING_DATA_UNSPECIFIED = 0
        EVALUATION_MISSING_DATA_INACTIVE = 1
        EVALUATION_MISSING_DATA_ACTIVE = 2
        EVALUATION_MISSING_DATA_NO_OP = 3
    aggregations = _messages.MessageField('Aggregation', 1, repeated=True)
    comparison = _messages.EnumField('ComparisonValueValuesEnum', 2)
    denominatorAggregations = _messages.MessageField('Aggregation', 3, repeated=True)
    denominatorFilter = _messages.StringField(4)
    duration = _messages.StringField(5)
    evaluationMissingData = _messages.EnumField('EvaluationMissingDataValueValuesEnum', 6)
    filter = _messages.StringField(7)
    forecastOptions = _messages.MessageField('ForecastOptions', 8)
    thresholdValue = _messages.FloatField(9)
    trigger = _messages.MessageField('Trigger', 10)