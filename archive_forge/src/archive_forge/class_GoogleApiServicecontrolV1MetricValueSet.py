from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1MetricValueSet(_messages.Message):
    """Represents a set of metric values in the same metric. Each metric value
  in the set should have a unique combination of start time, end time, and
  label values.

  Fields:
    metricName: The metric name defined in the service configuration.
    metricValues: The values in this metric.
  """
    metricName = _messages.StringField(1)
    metricValues = _messages.MessageField('GoogleApiServicecontrolV1MetricValue', 2, repeated=True)