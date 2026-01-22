from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DistributionCut(_messages.Message):
    """A DistributionCut defines a TimeSeries and thresholds used for measuring
  good service and total service. The TimeSeries must have ValueType =
  DISTRIBUTION and MetricKind = DELTA or MetricKind = CUMULATIVE. The computed
  good_service will be the estimated count of values in the Distribution that
  fall within the specified min and max.

  Fields:
    distributionFilter: A monitoring filter
      (https://cloud.google.com/monitoring/api/v3/filters) specifying a
      TimeSeries aggregating values. Must have ValueType = DISTRIBUTION and
      MetricKind = DELTA or MetricKind = CUMULATIVE.
    range: Range of values considered "good." For a one-sided range, set one
      bound to an infinite value.
  """
    distributionFilter = _messages.StringField(1)
    range = _messages.MessageField('GoogleMonitoringV3Range', 2)