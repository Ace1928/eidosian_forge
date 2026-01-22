from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricRange(_messages.Message):
    """A MetricRange is used when each window is good when the value x of a
  single TimeSeries satisfies range.min <= x <= range.max. The provided
  TimeSeries must have ValueType = INT64 or ValueType = DOUBLE and MetricKind
  = GAUGE.

  Fields:
    range: Range of values considered "good." For a one-sided range, set one
      bound to an infinite value.
    timeSeries: A monitoring filter
      (https://cloud.google.com/monitoring/api/v3/filters) specifying the
      TimeSeries to use for evaluating window quality.
  """
    range = _messages.MessageField('GoogleMonitoringV3Range', 1)
    timeSeries = _messages.StringField(2)