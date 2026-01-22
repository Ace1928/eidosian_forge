from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeSeriesRatio(_messages.Message):
    """A TimeSeriesRatio specifies two TimeSeries to use for computing the
  good_service / total_service ratio. The specified TimeSeries must have
  ValueType = DOUBLE or ValueType = INT64 and must have MetricKind = DELTA or
  MetricKind = CUMULATIVE. The TimeSeriesRatio must specify exactly two of
  good, bad, and total, and the relationship good_service + bad_service =
  total_service will be assumed.

  Fields:
    badServiceFilter: A monitoring filter
      (https://cloud.google.com/monitoring/api/v3/filters) specifying a
      TimeSeries quantifying bad service, either demanded service that was not
      provided or demanded service that was of inadequate quality. Must have
      ValueType = DOUBLE or ValueType = INT64 and must have MetricKind = DELTA
      or MetricKind = CUMULATIVE.
    goodServiceFilter: A monitoring filter
      (https://cloud.google.com/monitoring/api/v3/filters) specifying a
      TimeSeries quantifying good service provided. Must have ValueType =
      DOUBLE or ValueType = INT64 and must have MetricKind = DELTA or
      MetricKind = CUMULATIVE.
    totalServiceFilter: A monitoring filter
      (https://cloud.google.com/monitoring/api/v3/filters) specifying a
      TimeSeries quantifying total demanded service. Must have ValueType =
      DOUBLE or ValueType = INT64 and must have MetricKind = DELTA or
      MetricKind = CUMULATIVE.
  """
    badServiceFilter = _messages.StringField(1)
    goodServiceFilter = _messages.StringField(2)
    totalServiceFilter = _messages.StringField(3)